"""
ASR client for Zhipu GLM-ASR and SiliconFlow.
处理语音转录，支持长音频自动切分（30秒限制）
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import requests
import librosa
import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"zhipu", "siliconflow"}
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"

class ASRClient:
    """Client for ASR providers (Zhipu GLM-ASR, SiliconFlow)."""
    
    # API 限制：30秒
    MAX_DURATION_SEC = 30.0
    # 安全边界，留一点余量
    SAFE_DURATION_SEC = 28.0
    
    def __init__(self, api_key: str, provider: str = "zhipu"):
        """
        Initialize ASR client
        
        Args:
            api_key: ASR API key for the selected provider
        """
        provider = (provider or "zhipu").lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported ASR provider: {provider}")
        self.api_key = api_key
        self.provider = provider
        self.base_url = ZHIPU_BASE_URL if provider == "zhipu" else SILICONFLOW_BASE_URL
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def transcribe(
        self,
        audio_path: str,
        model_name: str = "glm-asr-2512",
        segment_as_token: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file (自动处理长音频切分)

        Args:
            audio_path: Path to audio file
            model_name: ASR model name (默认 glm-asr-2512)
            segment_as_token: 如果为 True，保持 segment 级别不分词（用于中文长文本）

        Returns:
            Dictionary with:
                - text: Full transcript text
                - tokens: List of tokens with word-level timestamps
                - confidence: List of confidence scores
        """
        self._segment_as_token = segment_as_token

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 加载音频获取时长
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        duration = len(y) / sr

        logger.info("Audio loaded: %.2fs duration, sr=%d", duration, sr)

        if duration <= self.SAFE_DURATION_SEC:
            # 短音频：直接转录
            return self._transcribe_segment(str(audio_path), model_name, y, sr, 0.0)
        else:
            # 长音频：切分后转录
            logger.info("Audio exceeds 30s limit, splitting...")
            return self._transcribe_long_audio(y, sr, model_name)
    
    def _transcribe_segment(
        self, 
        audio_path: str, 
        model_name: str,
        y: np.ndarray,
        sr: int,
        time_offset: float
    ) -> Dict[str, Any]:
        """转录单个音频片段"""
        with open(audio_path, 'rb') as f:
            files = {"file": (Path(audio_path).name, f, "audio/wav")}
            data = {"model": model_name}
            if self.provider == "zhipu":
                data["stream"] = "false"
            
            try:
                response = requests.post(
                    self.base_url,
                    data=data,
                    files=files,
                    headers=self.headers,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"ASR API request failed: {e}")
        
        # 解析响应
        return self._parse_response(result, y, sr, time_offset)
    
    def _transcribe_long_audio(
        self, 
        y: np.ndarray, 
        sr: int, 
        model_name: str
    ) -> Dict[str, Any]:
        """处理长音频：按静音点切分后分别转录"""
        # 找到所有可能的切分点（静音处）
        split_points = self._find_split_points(y, sr)
        
        # 生成片段
        segments = self._create_segments(y, sr, split_points)
        logger.info("Split into %d segments", len(segments))
        
        # 转录每个片段
        all_text = []
        all_tokens = []
        
        for idx, (seg_y, seg_start, seg_end) in enumerate(segments):
            logger.info("Transcribing segment %d/%d (%.2fs - %.2fs)", 
                       idx + 1, len(segments), seg_start, seg_end)
            
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, seg_y, sr)
            
            try:
                result = self._transcribe_segment(tmp_path, model_name, seg_y, sr, seg_start)
                all_text.append(result.get("text", ""))
                all_tokens.extend(result.get("tokens", []))
                # 每个片段转录后等待10秒，避免API速率限制
                if idx < len(segments) - 1:
                    logger.info("Waiting 10s before next segment...")
                    time.sleep(10)
            except Exception as e:
                logger.warning("Segment %d failed: %s", idx, e)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        
        # 合并结果，去除空格
        return {
            "text": "".join(all_text).replace(" ", ""),
            "tokens": all_tokens,
            "confidence": []
        }
    
    def _find_split_points(self, y: np.ndarray, sr: int) -> List[float]:
        """
        找到音频中的静音点作为潜在切分位置
        返回时间点列表（秒）
        """
        # 使用 librosa 检测非静音区间
        # top_db: 低于峰值多少 dB 认为是静音
        non_silent = librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=512)
        
        # 静音点 = 非静音区间之间的间隙
        split_points = []
        for i in range(len(non_silent) - 1):
            # 当前非静音段结束
            end_sample = non_silent[i][1]
            # 下一个非静音段开始
            next_start = non_silent[i + 1][0]
            
            # 取静音区间的中点作为切分点
            mid_sample = (end_sample + next_start) // 2
            split_time = mid_sample / sr
            split_points.append(split_time)
        
        return split_points
    
    def _create_segments(
        self, 
        y: np.ndarray, 
        sr: int, 
        split_points: List[float]
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        根据切分点创建音频片段，确保每段不超过30秒
        返回: [(音频数据, 开始时间, 结束时间), ...]
        """
        duration = len(y) / sr
        segments = []
        
        current_start = 0.0
        
        for split_time in split_points:
            segment_duration = split_time - current_start
            
            # 如果当前累积时长接近限制，在此处切分
            if segment_duration >= self.SAFE_DURATION_SEC:
                # 找到最后一个在限制内的切分点
                seg_end = split_time
                
                # 提取片段
                start_sample = int(current_start * sr)
                end_sample = int(seg_end * sr)
                seg_y = y[start_sample:end_sample]
                
                if len(seg_y) > 0:
                    segments.append((seg_y, current_start, seg_end))
                
                current_start = seg_end
        
        # 处理最后一段
        if current_start < duration:
            remaining = duration - current_start
            if remaining <= self.SAFE_DURATION_SEC:
                # 直接添加
                start_sample = int(current_start * sr)
                seg_y = y[start_sample:]
                if len(seg_y) > 0:
                    segments.append((seg_y, current_start, duration))
            else:
                # 最后一段也超长，强制按时间切分
                while current_start < duration:
                    seg_end = min(current_start + self.SAFE_DURATION_SEC, duration)
                    start_sample = int(current_start * sr)
                    end_sample = int(seg_end * sr)
                    seg_y = y[start_sample:end_sample]
                    if len(seg_y) > 0:
                        segments.append((seg_y, current_start, seg_end))
                    current_start = seg_end
        
        # 如果没有产生任何片段（音频太短或没有静音点），直接按时间强制切分
        if not segments:
            current_start = 0.0
            while current_start < duration:
                seg_end = min(current_start + self.SAFE_DURATION_SEC, duration)
                start_sample = int(current_start * sr)
                end_sample = int(seg_end * sr)
                seg_y = y[start_sample:end_sample]
                if len(seg_y) > 0:
                    segments.append((seg_y, current_start, seg_end))
                current_start = seg_end
        
        return segments

    @staticmethod
    def _normalize_time(value: Any, segment_duration: float) -> Optional[float]:
        """Normalize timestamp values; convert ms to seconds when needed."""
        try:
            t = float(value)
        except Exception:
            return None
        if segment_duration > 0 and t > max(segment_duration * 10.0, 100.0):
            t = t / 1000.0
        return t
    
    def _parse_response(
        self, 
        response: Dict[str, Any], 
        y: np.ndarray,
        sr: int,
        time_offset: float
    ) -> Dict[str, Any]:
        """
        Parse ASR API response and extract word-level timestamps
        
        Args:
            response: Raw API response
            y: Audio data (for duration estimation)
            sr: Sample rate
            time_offset: Time offset for this segment (seconds)
        
        Returns:
            Parsed result with text, tokens, confidence
        """
        result = {
            "text": "",
            "tokens": [],
            "confidence": []
        }
        
        duration = len(y) / sr
        
        # 响应格式兼容（智谱 / SiliconFlow）
        if isinstance(response, dict):
            # 获取转录文本
            text = response.get("text", "")
            if not text and "result" in response:
                text = response["result"].get("text", "")
            if not text and "data" in response:
                text = response["data"].get("text", "")

            # 检查是否有词级时间戳
            words_data = response.get("words", [])
            if not words_data and "result" in response:
                words_data = response["result"].get("words", [])

            # SiliconFlow 可能返回 segments
            segments = response.get("segments", [])
            if not segments and "result" in response:
                segments = response["result"].get("segments", [])
            if not segments and "data" in response:
                segments = response["data"].get("segments", [])

            if words_data:
                # 有词级时间戳
                for word_info in words_data:
                    start = self._normalize_time(word_info.get("start"), duration)
                    end = self._normalize_time(word_info.get("end"), duration)
                    if start is None or end is None:
                        continue
                    result["tokens"].append({
                        "word": word_info.get("word", word_info.get("text", "")),
                        "start": start + time_offset,
                        "end": end + time_offset,
                        "confidence": word_info.get("confidence", 0.9)
                    })
            elif segments:
                text_parts = []
                for seg in segments:
                    seg_text = (seg.get("text") or "").strip()
                    if seg_text:
                        text_parts.append(seg_text)
                    seg_words = seg.get("words") or seg.get("tokens") or []
                    if seg_words and not getattr(self, '_segment_as_token', False):
                        for word_info in seg_words:
                            start = self._normalize_time(word_info.get("start"), duration)
                            end = self._normalize_time(word_info.get("end"), duration)
                            if start is None or end is None:
                                continue
                            result["tokens"].append({
                                "word": word_info.get("word", word_info.get("text", "")),
                                "start": start + time_offset,
                                "end": end + time_offset,
                                "confidence": word_info.get("confidence", seg.get("confidence", 0.9)),
                            })
                    else:
                        seg_start = self._normalize_time(seg.get("start"), duration)
                        seg_end = self._normalize_time(seg.get("end"), duration)
                        if seg_text and seg_start is not None and seg_end is not None:
                            # segment_as_token 模式：整个 segment 作为一个 token（用于中文长文本）
                            if getattr(self, '_segment_as_token', False):
                                result["tokens"].append({
                                    "word": seg_text,
                                    "start": time_offset + seg_start,
                                    "end": time_offset + seg_end,
                                    "confidence": seg.get("confidence", 0.9),
                                })
                            else:
                                # 尝试按空格分词（英文），若无空格则按字符分词（中文）
                                words = seg_text.split()
                                if not words or (len(words) == 1 and len(seg_text) > 10):
                                    # 中文文本：按字符分词
                                    words = list(seg_text.replace(" ", ""))
                                if words:
                                    time_per_word = max(0.0, (seg_end - seg_start) / len(words))
                                    for i, word in enumerate(words):
                                        result["tokens"].append({
                                            "word": word,
                                            "start": time_offset + seg_start + i * time_per_word,
                                            "end": time_offset + seg_start + (i + 1) * time_per_word,
                                            "confidence": seg.get("confidence", 0.9),
                                        })
                if not text and text_parts:
                    text = " ".join(text_parts)

            # 去除空格
            result["text"] = text.replace(" ", "")

            if not result["tokens"]:
                # 无词级时间戳，根据文本均匀分配
                words = text.split()
                if not words or (len(words) == 1 and len(text) > 10):
                    # 中文文本：按字符分词
                    words = list(text.replace(" ", ""))
                if words:
                    time_per_word = duration / len(words)
                    for i, word in enumerate(words):
                        result["tokens"].append({
                            "word": word,
                            "start": time_offset + i * time_per_word,
                            "end": time_offset + (i + 1) * time_per_word,
                            "confidence": 0.9
                        })
        elif isinstance(response, str):
            # 纯文本响应，去除空格
            result["text"] = response.replace(" ", "")
            words = response.split()
            if not words or (len(words) == 1 and len(response) > 10):
                # 中文文本：按字符分词
                words = list(response.replace(" ", ""))
            if words:
                time_per_word = duration / len(words)
                for i, word in enumerate(words):
                    result["tokens"].append({
                        "word": word,
                        "start": time_offset + i * time_per_word,
                        "end": time_offset + (i + 1) * time_per_word,
                        "confidence": 0.9
                    })
        
        return result


__all__ = ["ASRClient"]
