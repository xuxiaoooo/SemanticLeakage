"""
CueGate Data Loader

从原始音频和cue标注中提取训练样本
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
from tqdm import tqdm

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "E-DAIC"
AGENT_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "outputs" / "E-DAIC"


# ============================================================================
# Helper Functions
# ============================================================================
def load_cue_annotations(cue_detection_path: Path) -> List[Dict]:
    """加载cue标注"""
    if not cue_detection_path.exists():
        return []
    with open(cue_detection_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('cues', [])


def load_kept_intervals(preprocess_spans_path: Path) -> List[Tuple[float, float]]:
    """加载保留的语音区间"""
    if not preprocess_spans_path.exists():
        return []
    with open(preprocess_spans_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(s, e) for s, e in data.get('kept_intervals', [])]


def create_frame_labels(
    num_samples: int,
    cue_intervals: List[Tuple[float, float]],
    frame_stride: int = 160,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    创建帧级标签
    
    Args:
        num_samples: 音频样本数
        cue_intervals: cue时间区间 [(start, end), ...]
        frame_stride: 帧步长（样本数）
        sample_rate: 采样率
    
    Returns:
        (num_frames,) 0/1标签
    """
    num_frames = num_samples // frame_stride
    labels = np.zeros(num_frames, dtype=np.float32)
    
    for start, end in cue_intervals:
        start_frame = int(start * sample_rate / frame_stride)
        end_frame = int(end * sample_rate / frame_stride) + 1
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        labels[start_frame:end_frame] = 1
    
    return labels


# ============================================================================
# Main Extraction Function
# ============================================================================
def extract_training_samples(
    audio: np.ndarray,
    cues: List[Dict],
    kept_intervals: List[Tuple[float, float]],
    sample_rate: int = 16000,
    segment_length: float = 3.0,
    context_ratio: float = 0.5,
    negative_ratio: float = 0.15,  # 降低负样本比例（正:负 约 6:1）
    frame_stride: int = 160,
    min_cue_ratio: float = 0.05,  # 正样本中cue至少占5%
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    从音频中提取训练样本
    
    策略：
    1. 对每个cue，提取包含该cue的片段（cue居中）
    2. 提取少量负样本（保持正负样本平衡）
    3. 对每个正样本进行多次随机偏移采样（数据增强）
    
    Args:
        audio: 原始音频波形
        cues: cue标注列表
        kept_intervals: 保留的语音区间
        sample_rate: 采样率
        segment_length: 每个片段长度（秒）
        context_ratio: 上下文比例（未使用）
        negative_ratio: 负样本比例（相对于正样本数量）
        frame_stride: 帧步长
        min_cue_ratio: 正样本中cue帧的最小占比
    
    Returns:
        List of (audio_segment, frame_labels)
    """
    samples = []
    segment_samples = int(segment_length * sample_rate)
    min_cue_frames = int(segment_samples // frame_stride * min_cue_ratio)
    
    if len(audio) < segment_samples:
        return samples
    
    # 提取cue区间
    cue_intervals = [(c['start'], c['end']) for c in cues]
    
    # 正样本：以cue为中心的片段，每个cue采样多次
    for cue in cues:
        cue_center = (cue['start'] + cue['end']) / 2
        cue_center_sample = int(cue_center * sample_rate)
        
        # 每个cue采样3次不同偏移
        for sample_idx in range(3):
            # 随机偏移，但确保cue在片段内
            cue_duration_samples = int((cue['end'] - cue['start']) * sample_rate)
            max_offset = segment_samples // 2 - cue_duration_samples // 2 - 100
            max_offset = max(0, max_offset)
            
            if sample_idx == 0:
                offset = 0  # 第一个居中
            else:
                offset = random.randint(-max_offset, max_offset) if max_offset > 0 else 0
            
            start = cue_center_sample - segment_samples // 2 + offset
            start = max(0, min(start, len(audio) - segment_samples))
            end = start + segment_samples
            
            segment = audio[start:end]
            
            # 创建相对于这个片段的帧标签
            segment_cue_intervals = []
            for cs, ce in cue_intervals:
                cs_sample = int(cs * sample_rate) - start
                ce_sample = int(ce * sample_rate) - start
                if ce_sample > 0 and cs_sample < segment_samples:
                    cs_rel = max(0, cs_sample) / sample_rate
                    ce_rel = min(segment_samples, ce_sample) / sample_rate
                    segment_cue_intervals.append((cs_rel, ce_rel))
            
            labels = create_frame_labels(
                segment_samples, segment_cue_intervals, frame_stride, sample_rate
            )
            
            # 确保正样本确实包含足够的cue帧
            if labels.sum() >= min_cue_frames:
                samples.append((segment, labels))
    
    # 负样本：不包含cue的区域（数量较少）
    # 基于原始cue数量，而不是已有样本数量
    num_positive = len(samples)
    num_negative = max(1, int(len(cues) * negative_ratio))
    
    # 找到非cue区域
    non_cue_regions = []
    if kept_intervals:
        for kept_start, kept_end in kept_intervals:
            region_start = kept_start
            for cue_start, cue_end in sorted(cue_intervals):
                if cue_end <= kept_start or cue_start >= kept_end:
                    continue
                if cue_start > region_start:
                    non_cue_regions.append((region_start, cue_start))
                region_start = max(region_start, cue_end)
            if region_start < kept_end:
                non_cue_regions.append((region_start, kept_end))
    else:
        # 没有kept_intervals，使用整个音频
        region_start = 0
        for cue_start, cue_end in sorted(cue_intervals):
            if cue_start > region_start:
                non_cue_regions.append((region_start, cue_start))
            region_start = max(region_start, cue_end)
        if region_start < len(audio) / sample_rate:
            non_cue_regions.append((region_start, len(audio) / sample_rate))
    
    # 过滤太短的区域
    min_region_length = segment_length + 0.5
    non_cue_regions = [(s, e) for s, e in non_cue_regions if e - s >= min_region_length]
    
    # 采样负样本
    for _ in range(num_negative):
        if not non_cue_regions:
            break
        
        region = random.choice(non_cue_regions)
        region_duration = region[1] - region[0]
        
        if region_duration < segment_length:
            continue
        
        # 随机选择起始位置
        max_start = region[1] - segment_length
        seg_start = random.uniform(region[0], max_start)
        
        start_sample = int(seg_start * sample_rate)
        end_sample = start_sample + segment_samples
        
        if end_sample > len(audio):
            continue
        
        segment = audio[start_sample:end_sample]
        labels = np.zeros(segment_samples // frame_stride, dtype=np.float32)
        
        samples.append((segment, labels))
    
    return samples


def load_all_training_data(
    segment_length: float = 3.0,
    sample_rate: int = 16000,
    frame_stride: int = 160,
    negative_ratio: float = 1.0,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    加载所有训练数据
    
    Args:
        segment_length: 片段长度（秒）
        sample_rate: 采样率
        frame_stride: 帧步长
        negative_ratio: 负样本比例
        verbose: 是否显示进度
    
    Returns:
        List of (audio_segment, frame_labels)
    """
    all_samples = []
    
    sample_dirs = sorted(AGENT_OUTPUTS_DIR.glob("*_AUDIO"))
    
    iterator = tqdm(sample_dirs, desc="Loading data") if verbose else sample_dirs
    
    for sample_dir in iterator:
        sample_id = sample_dir.name.replace("_AUDIO", "")
        
        # 音频路径
        audio_path = DATA_DIR / f"{sample_id}_P" / f"{sample_id}_AUDIO.wav"
        if not audio_path.exists():
            continue
        
        # 加载cue标注
        cue_path = sample_dir / "cue_detection.json"
        cues = load_cue_annotations(cue_path)
        
        if not cues:  # 跳过没有cue的样本
            continue
        
        # 加载保留区间
        preprocess_path = sample_dir / "preprocess_spans.json"
        kept_intervals = load_kept_intervals(preprocess_path)
        
        # 加载音频
        try:
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception:
            continue
        
        # 提取训练样本
        samples = extract_training_samples(
            audio, cues, kept_intervals,
            sample_rate=sample_rate,
            segment_length=segment_length,
            negative_ratio=negative_ratio,
            frame_stride=frame_stride,
        )
        
        all_samples.extend(samples)
    
    # 统计正负样本比例
    if all_samples and verbose:
        pos_frames = sum(labels.sum() for _, labels in all_samples)
        total_frames = sum(len(labels) for _, labels in all_samples)
        print(f"Total training samples: {len(all_samples)}")
        print(f"Positive frames: {pos_frames:.0f} / {total_frames:.0f} = {pos_frames/total_frames*100:.2f}%")
    
    return all_samples

