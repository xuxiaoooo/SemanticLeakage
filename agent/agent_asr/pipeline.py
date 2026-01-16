import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Allow running as script from any directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]  # project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import PACKAGE_ROOT as _AGENT_ROOT

from agent.config import settings
from agent.agent_asr.intake import IntakeProcessor
from agent.agent_asr.reconcile.proposer import ProposerAgent
from agent.agent_asr.reconcile.critic import CriticAgent
from agent.agent_asr.role import RoleAttributionAgent
from agent.agent_asr.transcript_builder import TranscriptMultiscaleBuilder
from agent.utils.asr_client import ASRClient
from agent.utils.llm_client import LLMClient
from agent.utils.audio_preprocessor import AudioPreprocessor
from agent.utils.asr_runner import ASRRunner
from agent.utils.token_alignment import align_evidence_map, filter_tokens_by_intervals
from agent.schemas import (
    FinalJSON, CriticOutput, InterviewerSegments, MultiScaleTranscript,
    EvidenceMapEntry, QATag, UncertainSpan, ChangelogEntry, Defect
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_to_project(path_like: Path) -> Path:
    """Resolve a path to an absolute location under the project root if not already absolute."""
    path_like = Path(path_like)
    return path_like if path_like.is_absolute() else (_PROJECT_ROOT / path_like)


# ---------------------------------------------------------------------------
# Checkpoint files expected for each step (relative to sample_dir)
# ---------------------------------------------------------------------------
CHECKPOINT_FILES = {
    "preprocess": "preprocess_spans.json",
    "step1": "step1_intake.json",
    "step2": "step2_reconcile.json",
    "step3": "step3_role.json",
    "step4": "transcript.multiscale.json",
}


class TranscriptPipeline:
    """Pipeline: preprocess -> ASR ensemble -> reconcile -> role -> multiscale transcript.

    Supports:
    - Multiple API keys for parallel ASR execution (pass list of keys)
    - Checkpoint/resume: skips steps whose JSON outputs already exist
    """

    def __init__(
        self,
        asr_api_key,  # str or List[str]
        llm_api_key=None,  # str or List[str]
        asr_runs: Optional[List[Dict[str, Any]]] = None,
        chunk_settings: Optional[Dict[str, Any]] = None,
        prompt_limits: Optional[Dict[str, Any]] = None,
        audio_preprocess_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Normalize keys to provider->list
        self.asr_api_keys: Dict[str, List[str]] = self._normalize_asr_keys(asr_api_key)
        self.llm_api_keys: List[str] = llm_api_key if isinstance(llm_api_key, list) else ([llm_api_key] if llm_api_key else [])

        # Create ASR clients by provider
        self.asr_clients_by_provider: Dict[str, List[ASRClient]] = {
            provider: [ASRClient(k, provider=provider) for k in keys if k]
            for provider, keys in self.asr_api_keys.items()
            if keys
        }
        self.asr_clients: List[ASRClient] = [
            client for clients in self.asr_clients_by_provider.values() for client in clients
        ]
        if not self.asr_clients:
            raise ValueError("No ASR API keys provided. Set ASR_API_KEYS or provider-specific keys.")
        self.default_asr_provider = self._pick_default_provider()
        # Use first LLM key for now (LLM calls are sequential)
        self.llm_client = LLMClient(self.llm_api_keys[0]) if self.llm_api_keys else None

        self.chunk_settings = {**settings.CHUNK_SETTINGS, **(chunk_settings or {})}
        self.prompt_limits = {**settings.PROMPT_LIMITS, **(prompt_limits or {})}

        self.intake_processor = IntakeProcessor()
        self.proposer = ProposerAgent(
            self.llm_client,
            chunk_settings=self.chunk_settings,
            prompt_limits=self.prompt_limits,
        )
        self.critic = CriticAgent(self.llm_client, prompt_limits=self.prompt_limits)
        self.role_attribution = RoleAttributionAgent()
        self.transcript_builder = TranscriptMultiscaleBuilder(self.llm_client)
        self.audio_preprocessor = AudioPreprocessor(**(audio_preprocess_kwargs or {}))

        self.asr_runs = asr_runs if asr_runs else settings.ASR_RUNS

    def _pick_default_provider(self) -> str:
        if self.asr_api_keys.get("zhipu"):
            return "zhipu"
        for provider, keys in self.asr_api_keys.items():
            if keys:
                return provider
        return "zhipu"

    @staticmethod
    def _normalize_asr_keys(asr_api_key) -> Dict[str, List[str]]:
        if isinstance(asr_api_key, dict):
            normalized: Dict[str, List[str]] = {}
            for provider, keys in asr_api_key.items():
                provider_key = (provider or "").lower()
                if not provider_key:
                    continue
                if isinstance(keys, list):
                    normalized[provider_key] = [k for k in keys if k]
                elif isinstance(keys, str):
                    normalized[provider_key] = [k.strip() for k in keys.split(",") if k.strip()]
                else:
                    normalized[provider_key] = []
            return normalized
        if isinstance(asr_api_key, list):
            return {"zhipu": [k for k in asr_api_key if k]}
        return {"zhipu": [asr_api_key] if asr_api_key else []}

    def _check_checkpoint(self, sample_dir: Optional[Path], step: str) -> bool:
        """Check if a checkpoint file exists for the given step."""
        if not sample_dir:
            return False
        ckpt_file = sample_dir / CHECKPOINT_FILES.get(step, "")
        return ckpt_file.exists()

    def _load_checkpoint(self, sample_dir: Path, step: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from JSON file."""
        ckpt_file = sample_dir / CHECKPOINT_FILES.get(step, "")
        if not ckpt_file.exists():
            return None
        with open(ckpt_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _run_asr_parallel(
        self, audio_path: str, max_workers: int = 3, segment_as_token: bool = False
    ) -> List[Dict[str, Any]]:
        """Run ASR with multiple API keys in parallel (round-robin assignment)."""
        outputs: List[Dict[str, Any]] = []
        seen_texts = set()

        # Build task list: (alias, model, idx, provider, client_idx)
        tasks_info = []
        client_indices = {provider: 0 for provider in self.asr_clients_by_provider.keys()}
        for run in self.asr_runs:
            model = run.get("model")
            repeat = max(1, int(run.get("repeat", 1)))
            alias = run.get("alias") or model
            provider = (run.get("provider") or self.default_asr_provider).lower()
            clients = self.asr_clients_by_provider.get(provider) or []
            if not clients:
                raise ValueError(f"No ASR clients available for provider: {provider}")
            for idx in range(repeat):
                # Round-robin assign to available clients
                client_idx = client_indices[provider] % len(clients)
                client_indices[provider] += 1
                tasks_info.append((alias, model, idx, provider, client_idx))

        def run_once(alias: str, model: str, idx: int, provider: str, client_idx: int):
            client = self.asr_clients_by_provider[provider][client_idx]
            result = client.transcribe(audio_path, model, segment_as_token=segment_as_token)
            text_key = (result.get("text") or "").strip()
            return alias, model, idx, provider, result, text_key

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_once, *info) for info in tasks_info]
            for fut in futures:
                alias, model, idx, provider, result, text_key = fut.result()
                duplicate = text_key in seen_texts
                if text_key:
                    seen_texts.add(text_key)
                outputs.append({
                    "id": f"{alias}{idx + 1}",
                    "model": model,
                    "provider": provider,
                    "result": result,
                    "duplicate": duplicate,
                    "score": ASRRunner.score_asr_output(result),
                })
        return outputs

    def process(
        self,
        audio_path: str,
        scenario_hint: str = "interview",
        diar_segments: Optional[List[Dict[str, Any]]] = None,
        output_dir: Optional[str] = None,
        save_intermediate: bool = True,
        enable_preprocess: bool = True,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        resume: bool = True,  # Enable checkpoint resume by default
        dataset: str = "edaic",  # 数据集类型，mandic 使用 segment 模式
    ) -> Dict[str, Any]:
        audio_path = _resolve_to_project(Path(audio_path))
        logger.info("Starting transcript pipeline for: %s", audio_path)

        sample_dir: Optional[Path] = None
        if output_dir:
            output_base = _resolve_to_project(Path(output_dir))
            sample_dir = output_base / audio_path.stem
            sample_dir.mkdir(parents=True, exist_ok=True)

        # Check if all steps are already complete
        if resume and sample_dir and self._check_checkpoint(sample_dir, "step4"):
            logger.info("All steps already complete (found %s), loading from checkpoint", CHECKPOINT_FILES["step4"])
            return self._load_full_result(sample_dir)

        working_audio_path = audio_path
        preprocess_report = None
        preprocess_result = None
        kept_intervals = None

        # ----- PREPROCESS (with checkpoint) -----
        if enable_preprocess:
            if resume and sample_dir and self._check_checkpoint(sample_dir, "preprocess"):
                logger.info("Preprocess checkpoint found, loading...")
                preprocess_report = self._load_checkpoint(sample_dir, "preprocess")
                kept_intervals = [tuple(iv) for iv in preprocess_report.get("kept_intervals", [])]
                # 有checkpoint但仍需要重新预处理以获得临时wav文件用于ASR
                try:
                    preprocess_kwargs = preprocess_kwargs or {}
                    preprocess_result = self.audio_preprocessor.preprocess(
                        audio_path=str(audio_path),
                        **preprocess_kwargs,
                    )
                    # 使用预处理后的临时wav文件进行ASR
                    working_audio_path = Path(preprocess_result.audio_path)
                    logger.info("Using preprocessed audio for ASR: %s", working_audio_path)
                except Exception as exc:
                    logger.warning("Audio preprocessing failed on resume, using original: %s", exc)
            else:
                try:
                    preprocess_kwargs = preprocess_kwargs or {}
                    preprocess_result = self.audio_preprocessor.preprocess(
                        audio_path=str(audio_path),
                        **preprocess_kwargs,
                    )
                    preprocess_report = preprocess_result.to_dict()
                    kept_intervals = preprocess_result.kept_intervals
                    # 使用预处理后的临时wav文件进行ASR
                    working_audio_path = Path(preprocess_result.audio_path)
                    logger.info(
                        "Audio preprocessing finished: %.2fs duration, temp wav: %s",
                        preprocess_result.original_duration, working_audio_path,
                    )
                    if save_intermediate and sample_dir:
                        self._save_json(sample_dir / CHECKPOINT_FILES["preprocess"], preprocess_report)
                except Exception as exc:
                    logger.warning("Audio preprocessing failed, continuing with original audio: %s", exc)
                    preprocess_report = None

        # ManDIC 使用 segment 模式（整句作为 token）
        segment_as_token = (dataset == "mandic")

        # ----- STEP 1: ASR (with checkpoint) -----
        if resume and sample_dir and self._check_checkpoint(sample_dir, "step1"):
            logger.info("Step 1 checkpoint found, loading intake data...")
            intake_output = self._load_checkpoint(sample_dir, "step1")
            top_asr = [{"id": intake_output["metadata"].get("asr_source", "A1"), "model": None}]
        else:
            logger.info(
                "Step 1: ASR ensemble (%d configs, %d clients across %d providers)%s",
                len(self.asr_runs),
                len(self.asr_clients),
                len(self.asr_clients_by_provider),
                " [segment_as_token mode]" if segment_as_token else "",
            )
            asr_outputs = self._run_asr_parallel(
                str(working_audio_path),
                max_workers=max(1, len(self.asr_clients)),
                segment_as_token=segment_as_token,
            )

            # cleanup temp enhanced audio after ASR runs
            if preprocess_result and getattr(preprocess_result, "temp_file", False):
                try:
                    Path(preprocess_result.audio_path).unlink(missing_ok=True)
                except Exception:
                    pass
            top_asr = ASRRunner.select_top(asr_outputs, k=2)
            if len(top_asr) < 2 and asr_outputs:
                top_asr = asr_outputs[:2]
            if not top_asr:
                raise RuntimeError("ASR failed: no hypotheses produced")

            asr_a_result = top_asr[0]["result"]
            asr_b_result = top_asr[1]["result"] if len(top_asr) > 1 else top_asr[0]["result"]
            logger.info(
                "Selected ASR: %s (%s tokens) & %s (%s tokens)",
                top_asr[0]["id"],
                len(asr_a_result.get("tokens", [])),
                top_asr[1]["id"] if len(top_asr) > 1 else top_asr[0]["id"],
                len(asr_b_result.get("tokens", [])),
            )

            intake_output = self.intake_processor.process(
                audio_path=str(working_audio_path),
                asr_a_result=asr_a_result,
                asr_b_result=asr_b_result,
                glossary=None,
                scenario_hint=scenario_hint,
            )
            # 只保留必要的元数据，不保存临时路径
            intake_output["metadata"]["asr_source"] = top_asr[0]["id"]
            intake_output["metadata"]["asr_model"] = top_asr[0].get("model")
            intake_output["metadata"]["asr_candidates"] = [
                {
                    "id": item["id"],
                    "model": item["model"],
                    "provider": item.get("provider"),
                    "score": item["score"],
                    "duplicate": item["duplicate"],
                    "token_count": len(item["result"].get("tokens", [])),
                }
                for item in asr_outputs
            ]
            # 只保存预处理的关键统计，不保存详细的 intervals
            if preprocess_report:
                intake_output["metadata"]["preprocess"] = {
                    "original_duration": preprocess_report.get("original_duration"),
                    "processed_duration": preprocess_report.get("processed_duration"),
                }

            if save_intermediate and sample_dir:
                self._save_json(sample_dir / CHECKPOINT_FILES["step1"], intake_output)

        # Filter tokens by kept_intervals for Step 2/3 (preserve originals for alignment)
        if not kept_intervals and preprocess_report:
            kept_intervals = [tuple(iv) for iv in preprocess_report.get("kept_intervals", [])]
        asr_a_filtered = self._filter_asr_data(intake_output["asr_a"], kept_intervals)
        asr_b_filtered = self._filter_asr_data(intake_output["asr_b"], kept_intervals)
        if kept_intervals:
            logger.info(
                "Filtered tokens by preprocess: A %d->%d, B %d->%d",
                len(intake_output["asr_a"]["tokens"]), len(asr_a_filtered["tokens"]),
                len(intake_output["asr_b"]["tokens"]), len(asr_b_filtered["tokens"]),
            )

        # ----- STEP 2: RECONCILE (with checkpoint) -----
        if resume and sample_dir and self._check_checkpoint(sample_dir, "step2"):
            logger.info("Step 2 checkpoint found, loading reconcile data...")
            step2_data = self._load_checkpoint(sample_dir, "step2")
            final_json = self._parse_final_json(step2_data["final"])
            critic_output = self._parse_critic_output(step2_data["critic"])
        else:
            logger.info("Step 2: Reconcile (proposer + critic)")
            max_revisions = 2
            for revision_attempt in range(max_revisions + 1):
                final_json = self.proposer.reconcile(
                    asr_a=asr_a_filtered,
                    asr_b=asr_b_filtered,
                    glossary=intake_output["glossary"],
                    scenario_hint=intake_output["scenario_hint"],
                )
                final_json = align_evidence_map(final_json, intake_output["asr_a"]["tokens"], intake_output["asr_b"]["tokens"])

                critic_output = self.critic.check(
                    final_json=final_json,
                    asr_a=intake_output["asr_a"],
                    asr_b=intake_output["asr_b"],
                )

                if critic_output.status == "pass":
                    logger.info("Critic passed on attempt %d", revision_attempt + 1)
                    break
                elif revision_attempt < max_revisions:
                    high_defects = [d for d in critic_output.defects if d.severity == "high"]
                    logger.warning(
                        "Critic requested revision (attempt %d/%d): %d high-severity defects",
                        revision_attempt + 1, max_revisions, len(high_defects)
                    )
                    defect_hints = "; ".join(d.description[:100] for d in high_defects[:3])
                    intake_output["scenario_hint"] = f"{scenario_hint} [FIX: {defect_hints}]"
                else:
                    logger.warning("Critic still reports issues after %d attempts, proceeding anyway", max_revisions)

            if save_intermediate and sample_dir:
                step2_payload = {
                    "final": final_json.model_dump(mode='json'),
                    "critic": critic_output.model_dump(mode='json'),
                }
                self._save_json(sample_dir / CHECKPOINT_FILES["step2"], step2_payload)

        # ----- STEP 3: ROLE ATTRIBUTION (with checkpoint) -----
        if resume and sample_dir and self._check_checkpoint(sample_dir, "step3"):
            logger.info("Step 3 checkpoint found, loading role data...")
            step3_data = self._load_checkpoint(sample_dir, "step3")
            interviewer_segments = InterviewerSegments(**step3_data)
        else:
            logger.info("Step 3: Role Attribution")
            interviewer_segments = self.role_attribution.attribute_role(
                final_json=final_json,
                asr_a=intake_output["asr_a"],
                asr_b=intake_output["asr_b"],
                diar_segments=diar_segments,
            )
            if save_intermediate and sample_dir:
                self._save_json(sample_dir / CHECKPOINT_FILES["step3"], interviewer_segments.model_dump(mode='json'))

        # ----- STEP 4: MULTISCALE (with checkpoint - final step) -----
        logger.info("Step 4: Multiscale transcripts")
        source_alias = intake_output["metadata"].get("asr_source", top_asr[0]["id"])
        # 精简 metadata，只保留必要信息，移除路径和详细预处理信息
        builder_metadata = {
            "asr_source": source_alias,
            "asr_model": intake_output["metadata"].get("asr_model", top_asr[0].get("model")),
            "token_count": len(intake_output["asr_a"]["tokens"]),
            "final_text_len": len(final_json.final_text or ""),
        }
        # 只保留预处理的关键统计信息，不保存详细的 intervals
        if preprocess_report:
            builder_metadata["preprocess"] = {
                "original_duration": preprocess_report.get("original_duration"),
                "processed_duration": preprocess_report.get("processed_duration"),
            }

        multiscale = self.transcript_builder.build(
            final_text=final_json.final_text,
            asr_tokens=intake_output["asr_a"]["tokens"],
            interviewer_segments=interviewer_segments,
            source_alias=source_alias,
            metadata=builder_metadata,
            allowed_intervals=kept_intervals,
            segment_mode=segment_as_token,  # ManDIC 使用 segment 模式
        )

        if save_intermediate and sample_dir:
            self._save_json(
                sample_dir / CHECKPOINT_FILES["step4"],
                multiscale.model_dump(mode='json'),
            )

        logger.info("Transcript pipeline completed successfully")
        return {
            "final_json": final_json,
            "critic_output": critic_output,
            "interviewer_segments": interviewer_segments,
            "multiscale": multiscale,
        }

    def _load_full_result(self, sample_dir: Path) -> Dict[str, Any]:
        """Load all results from checkpoints when pipeline is already complete."""
        step2_data = self._load_checkpoint(sample_dir, "step2")
        step3_data = self._load_checkpoint(sample_dir, "step3")
        step4_data = self._load_checkpoint(sample_dir, "step4")

        return {
            "final_json": self._parse_final_json(step2_data["final"]) if step2_data else None,
            "critic_output": self._parse_critic_output(step2_data["critic"]) if step2_data else None,
            "interviewer_segments": InterviewerSegments(**step3_data) if step3_data else None,
            "multiscale": MultiScaleTranscript(**step4_data) if step4_data else None,
        }

    @staticmethod
    def _parse_final_json(data: Dict[str, Any]) -> FinalJSON:
        """Parse FinalJSON from dict, handling nested Pydantic models."""
        return FinalJSON(
            final_text=data.get("final_text"),
            changed=data.get("changed", False),
            evidence_map=[EvidenceMapEntry(**e) for e in data.get("evidence_map", [])],
            uncertain_spans=[UncertainSpan(**u) for u in data.get("uncertain_spans", [])],
            changelog=[ChangelogEntry(**c) for c in data.get("changelog", [])],
            qa_tags=[QATag(**q) for q in data.get("qa_tags", [])],
        )

    @staticmethod
    def _parse_critic_output(data: Dict[str, Any]) -> CriticOutput:
        """Parse CriticOutput from dict."""
        return CriticOutput(
            defects=[Defect(**d) for d in data.get("defects", [])],
            status=data.get("status", "pass"),
        )

    @staticmethod
    def _filter_asr_data(
        asr_data: Dict[str, Any],
        kept_intervals: Optional[List[tuple]]
    ) -> Dict[str, Any]:
        """Filter ASR tokens/text to only include tokens within kept_intervals."""
        if not kept_intervals:
            return asr_data
        filtered_tokens = filter_tokens_by_intervals(asr_data.get("tokens", []), kept_intervals)
        filtered_text = " ".join(t.get("word", "") for t in filtered_tokens)
        return {**asr_data, "tokens": filtered_tokens, "text": filtered_text}

    @staticmethod
    def _save_json(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Batch processing CLI
# ---------------------------------------------------------------------------

def _collect_edaic_files(data_path: Path) -> List[Path]:
    """收集E-DAIC数据集的音频文件。"""
    audio_files = []
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir():
            audio_file = subdir / f"{subdir.name.replace('_P', '_AUDIO')}.wav"
            if audio_file.exists():
                audio_files.append(audio_file)
    return audio_files


def _collect_avec2014_files(data_path: Path) -> List[Path]:
    """收集AVEC2014数据集的视频文件（mp4）。"""
    video_files = []
    for split in ["train", "dev", "test"]:
        split_dir = data_path / split / "Freeform"
        if split_dir.exists():
            for video_file in sorted(split_dir.glob("*_Freeform_video.mp4")):
                video_files.append(video_file)
    return video_files


def _collect_mandic_files(data_path: Path) -> List[Path]:
    """收集ManDIC数据集的音频文件（WAV）。

    ManDIC数据集结构:
    - data/ManDIC/data/NN_XXXXXXXX_SXXX.WAV
    """
    audio_files = []
    data_dir = data_path / "data"
    if data_dir.exists():
        for audio_file in sorted(data_dir.glob("*.WAV")):
            audio_files.append(audio_file)
    return audio_files


def batch_process_daic(
    data_dir: str = "data/E-DAIC",
    output_dir: str = str(_AGENT_ROOT / "outputs/E-DAIC"),
    asr_api_keys: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    llm_api_keys: Optional[List[str]] = None,
    max_workers: int = 3,
    resume: bool = True,
    dataset: str = "edaic",
):
    """Batch process audio/video files with multi-threading.
    
    Args:
        data_dir: 数据目录路径
        output_dir: 输出目录路径
        asr_api_keys: ASR API keys列表或provider->keys映射
        llm_api_keys: LLM API keys列表
        max_workers: 并行worker数量
        resume: 是否继续之前的进度
        dataset: 数据集类型 "edaic" 或 "avec2014"
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _PROJECT_ROOT / data_path
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # 根据数据集类型收集文件
    if dataset.lower() == "avec2014":
        audio_files = _collect_avec2014_files(data_path)
        logger.info("Dataset: AVEC2014 (video files)")
    elif dataset.lower() == "mandic":
        audio_files = _collect_mandic_files(data_path)
        logger.info("Dataset: ManDIC (audio files)")
    else:
        audio_files = _collect_edaic_files(data_path)
        logger.info("Dataset: E-DAIC (audio files)")

    logger.info("Found %d audio files in %s", len(audio_files), data_dir)

    # Load API keys from env if not provided
    if not asr_api_keys:
        asr_keys_str = os.environ.get("ASR_API_KEYS", os.environ.get("ASR_API_KEY", ""))
        asr_api_keys = [k.strip() for k in asr_keys_str.split(",") if k.strip()]
    if not llm_api_keys:
        llm_keys_str = os.environ.get("LLM_API_KEYS", os.environ.get("LLM_API_KEY", ""))
        llm_api_keys = [k.strip() for k in llm_keys_str.split(",") if k.strip()]

    if isinstance(asr_api_keys, dict):
        has_any = any(keys for keys in asr_api_keys.values())
        if not has_any:
            raise ValueError("No ASR API keys provided. Set ASR_API_KEYS env var or pass --asr-keys")
    elif not asr_api_keys:
        raise ValueError("No ASR API keys provided. Set ASR_API_KEYS env var or pass --asr-keys")
    if not llm_api_keys:
        raise ValueError("No LLM API keys provided. Set LLM_API_KEYS env var or pass --llm-keys")

    # Split API keys across workers (round-robin)
    def create_pipeline_for_worker(worker_id: int) -> TranscriptPipeline:
        if isinstance(asr_api_keys, dict):
            selected_keys: Dict[str, List[str]] = {}
            for provider, keys in asr_api_keys.items():
                if keys:
                    selected_keys[provider] = [keys[worker_id % len(keys)]]
            asr_key = selected_keys
        else:
            asr_key = asr_api_keys[worker_id % len(asr_api_keys)]
        llm_key = llm_api_keys[worker_id % len(llm_api_keys)]
        return TranscriptPipeline(asr_api_key=asr_key, llm_api_key=llm_key)

    # Process single file
    def process_one(audio_path: Path, worker_id: int) -> dict:
        pipeline = create_pipeline_for_worker(worker_id)
        sample_name = audio_path.stem
        try:
            result = pipeline.process(
                audio_path=str(audio_path),
                output_dir=str(output_path),
                resume=resume,
                dataset=dataset,  # 传递数据集类型
            )
            return {"sample": sample_name, "status": "success"}
        except Exception as e:
            logger.error("Failed to process %s: %s", sample_name, e)
            return {"sample": sample_name, "status": "error", "error": str(e)}

    # Run with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_audio = {
            executor.submit(process_one, audio, idx % max_workers): audio
            for idx, audio in enumerate(audio_files)
        }
        for future in as_completed(future_to_audio):
            audio = future_to_audio[future]
            result = future.result()
            results.append(result)
            done = len(results)
            logger.info("Progress: %d/%d - %s: %s", done, len(audio_files), result["sample"], result["status"])

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    errors = [r for r in results if r["status"] == "error"]
    logger.info("Batch complete: %d/%d success, %d errors", success, len(audio_files), len(errors))
    if errors:
        for e in errors:
            logger.warning("  - %s: %s", e["sample"], e.get("error", ""))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Pipeline for depression datasets")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="edaic",
        choices=["edaic", "avec2014", "mandic"],
        help="数据集类型: edaic (E-DAIC), avec2014 (AVEC2014) 或 mandic (ManDIC)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="并行worker数量 (默认: 3)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传，重新处理所有文件"
    )
    args = parser.parse_args()

    # 根据数据集设置路径
    if args.dataset == "avec2014":
        DATA_DIR = "data/AVEC2014"
        OUTPUT_DIR = str(_AGENT_ROOT / "outputs/AVEC2014")
    elif args.dataset == "mandic":
        DATA_DIR = "data/ManDIC"
        OUTPUT_DIR = str(_AGENT_ROOT / "outputs/ManDIC")
    else:
        DATA_DIR = "data/E-DAIC"
        OUTPUT_DIR = str(_AGENT_ROOT / "outputs/E-DAIC")

    MAX_WORKERS = args.workers

    # 从 api_keys.json 或环境变量加载 API keys
    api_keys = settings.load_api_keys()
    asr_keys = {
        "zhipu": api_keys.get("ZHIPU_API_KEY", []),
        "siliconflow": api_keys.get("SILICONFLOW_API_KEY", []),
    }
    llm_keys = api_keys.get("DEEPSEEK_API_KEY", [])

    if not any(asr_keys.values()):
        raise ValueError("未找到 ASR API key，请在 agent/config/api_keys.json 中配置 ZHIPU_API_KEY 或 SILICONFLOW_API_KEY")
    if not llm_keys:
        raise ValueError("未找到 LLM API key，请在 agent/config/api_keys.json 中配置 DEEPSEEK_API_KEY")

    logger.info(
        "配置: dataset=%s, %d ZHIPU ASR keys, %d SILICONFLOW ASR keys, %d LLM keys, %d workers",
        args.dataset,
        len(asr_keys.get("zhipu", [])),
        len(asr_keys.get("siliconflow", [])),
        len(llm_keys),
        MAX_WORKERS,
    )

    batch_process_daic(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        asr_api_keys=asr_keys,
        llm_api_keys=llm_keys,
        max_workers=MAX_WORKERS,
        resume=not args.no_resume,
        dataset=args.dataset,
    )
