"""
Cue Detection Pipeline: Main entry point for Agent Cue.
使用LLM直接检测抑郁相关词。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Allow running as script from any directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import PACKAGE_ROOT as _AGENT_ROOT

from agent.schemas import (
    MultiScaleTranscript,
    CueItem,
    CueStatistics,
    CueDetectionResult,
)
from agent.agent_cue.detector import CueDetector
from agent.utils.llm_client import LLMClient
from agent.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_path(path_like: Path, base: Path) -> Path:
    """Resolve to absolute path using base when a relative path is provided."""
    path_like = Path(path_like)
    return path_like if path_like.is_absolute() else (base / path_like)


class CueDetectionPipeline:
    """Pipeline for detecting depression-related cues in transcripts."""

    def __init__(self, llm_client: Optional[LLMClient] = None, enable_review: bool = True):
        """
        Initialize pipeline.

        Args:
            llm_client: LLM client for cue detection. Required.
            enable_review: 是否启用二次审核过滤（默认启用）
        """
        if llm_client is None:
            raise ValueError("LLM client is required for cue detection")
        self.llm_client = llm_client
        self.enable_review = enable_review
        self.detector = CueDetector(llm_client, enable_review=enable_review)

    def _load_transcript(self, path: Path) -> MultiScaleTranscript:
        """Load transcript from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MultiScaleTranscript(**data)

    def _compute_statistics(
        self,
        cues: List[Dict[str, Any]],
        total_words: int,
        total_duration: float,
    ) -> CueStatistics:
        """Compute detection statistics."""
        cue_time = sum(item["end"] - item["start"] for item in cues)

        return CueStatistics(
            total_words=total_words,
            cues_detected=len(cues),
            cue_time_coverage_sec=round(cue_time, 2),
            cue_time_coverage_ratio=round(cue_time / total_duration, 4) if total_duration > 0 else 0.0,
        )

    def run(
        self,
        transcript_path: Path,
        output_path: Optional[Path] = None,
    ) -> CueDetectionResult:
        """
        Run cue detection pipeline.

        Args:
            transcript_path: Path to transcript.multiscale.json
            output_path: Optional path to save cue_detection.json

        Returns:
            CueDetectionResult with detected cues
        """
        transcript_path = _resolve_path(transcript_path, _PROJECT_ROOT)
        if output_path:
            output_path = _resolve_path(output_path, _PROJECT_ROOT)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading transcript from %s", transcript_path)
        transcript = self._load_transcript(transcript_path)
        words = transcript.words
        sentences = transcript.sentences

        # ManDIC segment 模式：words 为空，使用 sentences
        segment_mode = (not words) and sentences

        if segment_mode:
            logger.info("Segment mode: detecting cues in %d sentences", len(sentences))
            detected = self.detector.detect_from_sentences(sentences)
            total_duration = sentences[-1].end if sentences else 0.0
            total_items = len(sentences)

            # 构建CueItem列表（使用 sentence_id）
            cue_items = [
                CueItem(
                    id=idx,
                    word_id=item["sentence_id"],  # 在 segment 模式下是 sentence_id
                    text=item["text"],
                    start=item["start"],
                    end=item["end"],
                    speaker=item["speaker"],
                )
                for idx, item in enumerate(detected)
            ]
        else:
            if not words:
                logger.warning("No words in transcript, returning empty result")
                return CueDetectionResult()

            # 默认模式：基于 words 检测
            logger.info("Detecting depression cues in %d words using LLM", len(words))
            detected = self.detector.detect(words)
            total_duration = words[-1].end if words else 0.0
            total_items = len(words)

            # 构建CueItem列表
            cue_items = [
                CueItem(
                    id=idx,
                    word_id=item["word_id"],
                    text=item["text"],
                    start=item["start"],
                    end=item["end"],
                    speaker=item["speaker"],
                )
                for idx, item in enumerate(detected)
            ]

        # 构建结果
        result = CueDetectionResult(
            metadata={
                "llm_model": self.llm_client.model,
                "total_words_scanned": total_items,
                "review_enabled": self.enable_review,
                "segment_mode": segment_mode,
            },
            cues=cue_items,
            statistics=self._compute_statistics(detected, total_items, total_duration),
        )

        # 保存结果
        if output_path:
            logger.info("Saving result to %s", output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        return result


# ---------------------------------------------------------------------------
# Batch processing function
# ---------------------------------------------------------------------------
def batch_process_outputs(
    outputs_dir: str = str(_AGENT_ROOT / "outputs"),
    llm_api_key: Optional[str] = None,
    force: bool = True,
    max_workers: int = 1,
    enable_review: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch process all samples in outputs directory.

    Args:
        outputs_dir: Path to outputs directory
        llm_api_key: LLM API key for detection
        force: Overwrite existing cue_detection.json (default: True, always overwrite)
        max_workers: Number of parallel workers (currently sequential)
        enable_review: 是否启用二次审核过滤（默认启用）

    Returns:
        List of processing results
    """
    outputs_path = _resolve_path(outputs_dir, _PROJECT_ROOT)
    if not outputs_path.exists():
        logger.error("Outputs directory not found: %s", outputs_path)
        return []

    # Find all samples with transcript.multiscale.json
    samples = []
    for transcript_path in sorted(outputs_path.glob("**/transcript.multiscale.json")):
        sample_dir = transcript_path.parent
        # Accept all directories with transcript.multiscale.json
        # (E-DAIC uses *_AUDIO, AVEC2014 uses *_Freeform_video)
        samples.append(sample_dir)

    logger.info("Found %d samples to process in %s", len(samples), outputs_path)
    logger.info("Review mode: %s", "ENABLED" if enable_review else "DISABLED")

    if not samples:
        logger.warning("No samples found with transcript.multiscale.json")
        return []

    # Initialize pipeline
    llm_client = LLMClient(llm_api_key)
    pipeline = CueDetectionPipeline(llm_client=llm_client, enable_review=enable_review)

    # Process each sample
    results = []
    for idx, sample_dir in enumerate(samples, 1):
        transcript_path = sample_dir / "transcript.multiscale.json"
        output_path = sample_dir / "cue_detection.json"
        sample_name = sample_dir.name

        # Always process (force mode) - overwrite existing files
        # No skipping logic - user may run multiple times until satisfied

        # Process
        logger.info("[%d/%d] Processing %s...", idx, len(samples), sample_name)
        try:
            result = pipeline.run(
                transcript_path=transcript_path,
                output_path=output_path
            )
            logger.info(
                "  ✓ %s: %d cues detected",
                sample_name,
                result.statistics.cues_detected,
            )
            results.append({
                "sample": sample_name,
                "status": "success",
                "cues_detected": result.statistics.cues_detected,
            })
        except Exception as e:
            logger.error("  ✗ %s failed: %s", sample_name, e, exc_info=True)
            results.append({
                "sample": sample_name,
                "status": "error",
                "error": str(e)
            })

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    errors = [r for r in results if r["status"] == "error"]

    logger.info(
        "\n=== Batch Processing Summary ===\n"
        "Total: %d\n"
        "Success: %d\n"
        "Errors: %d",
        len(samples), success, len(errors)
    )

    if errors:
        logger.warning("Failed samples:")
        for e in errors:
            logger.warning("  - %s: %s", e["sample"], e.get("error", ""))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cue Detection Pipeline for depression datasets")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="edaic",
        choices=["edaic", "avec2014", "mandic"],
        help="数据集类型: edaic (E-DAIC), avec2014 (AVEC2014) 或 mandic (ManDIC)"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="禁用二次审核过滤"
    )
    args = parser.parse_args()

    # 根据数据集设置输出目录
    if args.dataset == "avec2014":
        OUTPUTS_DIR = str(_AGENT_ROOT / "outputs/AVEC2014")
    elif args.dataset == "mandic":
        OUTPUTS_DIR = str(_AGENT_ROOT / "outputs/ManDIC")
    else:
        OUTPUTS_DIR = str(_AGENT_ROOT / "outputs/E-DAIC")

    FORCE = True  # 默认覆盖已存在的结果，可多次运行直到满意
    ENABLE_REVIEW = not args.no_review  # 默认启用二次审核过滤

    # 从 api_keys.json 加载 LLM API key
    api_keys = settings.load_api_keys()
    llm_keys = api_keys.get("DEEPSEEK_API_KEY", [])

    if not llm_keys:
        raise ValueError("未找到 LLM API key，请在 agent/config/api_keys.json 中配置 DEEPSEEK_API_KEY")

    llm_key = llm_keys[0]

    logger.info("配置: dataset=%s, outputs_dir=%s, force=%s, review=%s", 
                args.dataset, OUTPUTS_DIR, FORCE, ENABLE_REVIEW)

    batch_process_outputs(
        outputs_dir=OUTPUTS_DIR,
        llm_api_key=llm_key,
        force=FORCE,
        enable_review=ENABLE_REVIEW,
    )


__all__ = ["CueDetectionPipeline", "batch_process_outputs"]
