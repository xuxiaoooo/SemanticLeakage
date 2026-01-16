"""
CueGate Inference: 即插即用的Cue检测接口

使用方式：
    from CueGate.CueGate.inference import CueDetector
    
    # 加载模型
    detector = CueDetector("path/to/checkpoint.pt")
    
    # 检测cue
    results = detector.detect("path/to/audio.wav")
    # results: [{'start': 1.2, 'end': 1.8, 'score': 0.95}, ...]
    
    # 或者直接传入波形
    results = detector.detect(waveform, sample_rate=16000)
"""

import json
from pathlib import Path
from typing import List, Dict, Union, Optional

import librosa
import numpy as np
import torch

from model import CueGate, CueGateConfig


class CueDetector:
    """
    CueGate即插即用检测器
    
    加载训练好的权重，输入任意音频，输出cue的时间span
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.15,  # 降低默认阈值，因为模型输出概率偏低
    ):
        """
        Args:
            checkpoint_path: 模型权重路径，None则使用随机初始化
            device: 设备，None则自动选择
            threshold: cue检测阈值
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        
        # 加载模型
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            config = checkpoint.get('config', CueGateConfig())
            self.model = CueGate(config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")
            if 'val_metrics' in checkpoint:
                metrics = checkpoint['val_metrics']
                print(f"  Val F1: {metrics.get('f1', 'N/A'):.4f}")
        else:
            self.model = CueGate(CueGateConfig()).to(self.device)
            if checkpoint_path:
                print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
        
        self.model.eval()
        self.sample_rate = self.model.config.sample_rate
    
    def detect(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        threshold: Optional[float] = None,
        min_duration: float = 0.1,
        merge_gap: float = 0.05,
    ) -> List[Dict]:
        """
        检测音频中的cue片段
        
        Args:
            audio: 音频文件路径 或 波形数组 (samples,) 或 (channels, samples)
            sample_rate: 如果audio是数组，需要指定采样率
            threshold: cue概率阈值（None则使用默认值）
            min_duration: 最小cue时长（秒）
            merge_gap: 合并间隔小于此值的相邻片段
        
        Returns:
            List of dicts: [{'start': float, 'end': float, 'score': float}, ...]
        """
        threshold = threshold if threshold is not None else self.threshold
        
        # 加载/预处理音频
        waveform = self._load_audio(audio, sample_rate)
        
        # 检测
        results = self.model.detect(
            waveform,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )
        
        return results
    
    def detect_with_probs(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> Dict:
        """
        检测并返回帧级概率
        
        Returns:
            {
                'cue_probs': np.ndarray (num_frames,),
                'frame_times': np.ndarray (num_frames,),
                'gate': np.ndarray (num_frames,),
            }
        """
        waveform = self._load_audio(audio, sample_rate)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(waveform)
        
        num_frames = outputs['num_frames']
        frame_stride = self.model.frame_stride
        
        frame_times = np.arange(num_frames) * frame_stride / self.sample_rate
        
        return {
            'cue_probs': outputs['cue_probs'][0].cpu().numpy(),
            'frame_times': frame_times,
            'gate': outputs['gate'][0].cpu().numpy(),
        }
    
    def _load_audio(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """加载并预处理音频"""
        if isinstance(audio, (str, Path)):
            waveform, sr = librosa.load(audio, sr=self.sample_rate, mono=True)
            waveform = torch.FloatTensor(waveform)
        elif isinstance(audio, np.ndarray):
            if sample_rate and sample_rate != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            if audio.ndim > 1:
                audio = audio.mean(axis=0)  # 转单声道
            waveform = torch.FloatTensor(audio)
        elif isinstance(audio, torch.Tensor):
            waveform = audio.float()
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        # 归一化
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        
        return waveform
    
    def batch_detect(
        self,
        audio_paths: List[Union[str, Path]],
        threshold: Optional[float] = None,
        min_duration: float = 0.1,
        merge_gap: float = 0.05,
    ) -> Dict[str, List[Dict]]:
        """
        批量检测多个音频文件
        
        Returns:
            {audio_path: [cue_spans], ...}
        """
        results = {}
        for path in audio_paths:
            try:
                cues = self.detect(
                    path,
                    threshold=threshold,
                    min_duration=min_duration,
                    merge_gap=merge_gap,
                )
                results[str(path)] = cues
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results[str(path)] = []
        
        return results
    
    def save_results(
        self,
        results: List[Dict],
        output_path: Union[str, Path],
        audio_path: Optional[str] = None,
    ):
        """保存检测结果到JSON"""
        output = {
            'audio_path': audio_path,
            'threshold': self.threshold,
            'cues': results,
            'num_cues': len(results),
            'total_cue_duration': sum(c['end'] - c['start'] for c in results),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


def load_detector(checkpoint_path: Optional[str] = None, **kwargs) -> CueDetector:
    """快捷函数：加载CueDetector"""
    if checkpoint_path is None:
        # 使用默认路径
        default_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
        if default_path.exists():
            checkpoint_path = str(default_path)
    
    return CueDetector(checkpoint_path, **kwargs)


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect cues in audio files")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.15, help="Detection threshold")
    parser.add_argument("--min-duration", type=float, default=0.1, help="Minimum cue duration")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    # 加载检测器
    detector = load_detector(args.checkpoint, threshold=args.threshold)
    
    # 检测
    print(f"\nDetecting cues in: {args.audio}")
    results = detector.detect(
        args.audio,
        threshold=args.threshold,
        min_duration=args.min_duration,
    )
    
    # 输出结果
    print(f"\nFound {len(results)} cue(s):")
    for i, cue in enumerate(results, 1):
        print(f"  {i}. [{cue['start']:.3f}s - {cue['end']:.3f}s] score={cue['score']:.3f}")
    
    # 保存
    if args.output:
        detector.save_results(results, args.output, audio_path=args.audio)
        print(f"\nResults saved to: {args.output}")

