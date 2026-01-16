"""
E-DAIC 等长切割对比实验
========================

功能：
1. 从原始音频中读取，按等长（默认30s）切割成多个样本
2. 支持两种variant：pre（预处理后）和 cue-exclude（去掉cue后）
3. 在内存中完成切割，不生成中间文件
4. 跑4个baseline模型：DepAudioNet / ABAFnet / DALF / STFN
5. 5次随机种子重复，输出MAE±std和RMSE±std

使用方式：
    python CueGate/Baseline/run_segment_experiments.py
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# ============================================================================
# Path Configuration
# ============================================================================
BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "E-DAIC"
AGENT_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "outputs" / "E-DAIC"

# 确保可以导入各模型的模块
sys.path.insert(0, str(BASELINE_ROOT / "DepAudioNet"))
sys.path.insert(0, str(BASELINE_ROOT / "ABAFnet"))
sys.path.insert(0, str(BASELINE_ROOT / "DALF"))
sys.path.insert(0, str(BASELINE_ROOT / "STFN"))

from DALF.train import train_one_epoch, evaluate as dalf_evaluate
from DALF.train import DALFNet, DALFConfig
from STFN.train import train_model as train_stfn_model, STFNModel


# ============================================================================
# Audio Processing Utilities
# ============================================================================
def load_cue_intervals(cue_detection_path: Path) -> List[Tuple[float, float]]:
    """从cue_detection.json获取cue时间区间"""
    if not cue_detection_path.exists():
        return []
    with open(cue_detection_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(cue['start'], cue['end']) for cue in data.get('cues', [])]


def load_kept_intervals(preprocess_spans_path: Path) -> List[Tuple[float, float]]:
    """从preprocess_spans.json获取保留的音频区间"""
    if not preprocess_spans_path.exists():
        return []
    with open(preprocess_spans_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(s, e) for s, e in data.get('kept_intervals', [])]


def extract_intervals(audio: np.ndarray, sr: int, intervals: List[Tuple[float, float]]) -> np.ndarray:
    """从音频中提取指定区间并拼接"""
    if not intervals:
        return audio
    segments = []
    for start, end in intervals:
        s = max(0, int(start * sr))
        e = min(len(audio), int(end * sr))
        if e > s:
            segments.append(audio[s:e])
    return np.concatenate(segments) if segments else np.array([], dtype=audio.dtype)


def remove_intervals(audio: np.ndarray, sr: int, intervals: List[Tuple[float, float]]) -> np.ndarray:
    """从音频中移除指定区间"""
    if not intervals:
        return audio
    mask = np.ones(len(audio), dtype=bool)
    for start, end in intervals:
        s = max(0, int(start * sr))
        e = min(len(audio), int(end * sr))
        mask[s:e] = False
    return audio[mask]


def segment_audio_fixed_length(audio: np.ndarray, segment_samples: int) -> List[np.ndarray]:
    """将音频切割成固定长度片段，不足的丢弃"""
    if len(audio) < segment_samples:
        return []
    num_segments = len(audio) // segment_samples
    return [audio[i*segment_samples:(i+1)*segment_samples] for i in range(num_segments)]


# ============================================================================
# Data Loading - In-Memory Processing
# ============================================================================
def load_all_segments(
    variant: str = "pre",
    segment_length: float = 30.0,
    sample_rate: int = 16000,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    加载所有样本并切割成等长片段（在内存中完成）
    
    Args:
        variant: "pre" 或 "cue-exclude"
        segment_length: 每个片段长度（秒）
        sample_rate: 采样率
    
    Returns:
        (audio_segments, labels, sample_ids)
    """
    segment_samples = int(segment_length * sample_rate)
    
    # 加载标签
    labels_path = DATA_DIR / "Detailed_PHQ8_Labels.csv"
    labels_df = pd.read_csv(labels_path)
    labels_dict = dict(zip(labels_df['Participant_ID'], labels_df['PHQ_8Total']))
    
    all_segments = []
    all_labels = []
    all_sample_ids = []
    
    # 遍历所有样本
    sample_dirs = sorted(AGENT_OUTPUTS_DIR.glob("*_AUDIO"))
    
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name.replace("_AUDIO", "")
        sample_id_int = int(sample_id)
        
        if sample_id_int not in labels_dict:
            continue
        
        label = labels_dict[sample_id_int]
        
        # 原始音频路径
        audio_path = DATA_DIR / f"{sample_id}_P" / f"{sample_id}_AUDIO.wav"
        if not audio_path.exists():
            continue
        
        # 加载音频
        try:
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception:
            continue
        
        # 获取预处理后的保留区间
        preprocess_path = sample_dir / "preprocess_spans.json"
        kept_intervals = load_kept_intervals(preprocess_path)
        
        # 提取保留区间作为"pre"版本
        if kept_intervals:
            pre_audio = extract_intervals(audio, sr, kept_intervals)
        else:
            pre_audio = audio
        
        if variant == "pre":
            target_audio = pre_audio
        else:  # cue-exclude
            # 获取cue区间
            cue_path = sample_dir / "cue_detection.json"
            cue_intervals = load_cue_intervals(cue_path)
            
            if cue_intervals and kept_intervals:
                # 从kept区间的音频中移除cue
                # 需要将cue时间映射到提取后的时间线
                target_audio = _remove_cues_from_kept(audio, sr, kept_intervals, cue_intervals)
            elif cue_intervals:
                target_audio = remove_intervals(pre_audio, sr, cue_intervals)
            else:
                target_audio = pre_audio
        
        # 切割成等长片段
        segments = segment_audio_fixed_length(target_audio, segment_samples)
        
        for seg in segments:
            all_segments.append(seg)
            all_labels.append(label)
            all_sample_ids.append(sample_id)
    
    return all_segments, np.array(all_labels, dtype=np.float32), all_sample_ids


def _remove_cues_from_kept(
    audio: np.ndarray,
    sr: int,
    kept_intervals: List[Tuple[float, float]],
    cue_intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """从保留区间的音频中移除cue"""
    segments = []
    
    for kept_start, kept_end in kept_intervals:
        ks = max(0, int(kept_start * sr))
        ke = min(len(audio), int(kept_end * sr))
        segment = audio[ks:ke]
        
        # 检查哪些cue落在这个区间
        cues_in_seg = []
        for cs, ce in cue_intervals:
            if ce > kept_start and cs < kept_end:
                rel_s = max(0, cs - kept_start)
                rel_e = min(kept_end - kept_start, ce - kept_start)
                cues_in_seg.append((rel_s, rel_e))
        
        if cues_in_seg:
            segment = remove_intervals(segment, sr, cues_in_seg)
        
        if len(segment) > 0:
            segments.append(segment)
    
    return np.concatenate(segments) if segments else np.array([], dtype=audio.dtype)


# ============================================================================
# Dataset and DataLoader
# ============================================================================
class FixedSegmentDataset(Dataset):
    """固定长度音频片段数据集"""
    
    def __init__(self, segments: List[np.ndarray], labels: np.ndarray, is_training: bool = True):
        self.segments = segments
        self.labels = labels
        self.is_training = is_training
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        audio = torch.FloatTensor(self.segments[idx])
        label = torch.FloatTensor([self.labels[idx]])
        
        # 归一化
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()
        
        # 数据增强（仅训练时）
        if self.is_training:
            gain = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            audio = audio * gain
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise
        
        # 添加通道维度
        audio = audio.unsqueeze(0)
        
        return audio, label


class SegmentDataManager:
    """数据管理器：负责加载、划分和创建DataLoader"""
    
    def __init__(
        self,
        variant: str = "pre",
        segment_length: float = 30.0,
        sample_rate: int = 16000,
        batch_size: int = 16,
        num_workers: int = 4,
        random_state: int = 42,
        val_size: float = 0.2,
        test_size: float = 0.1,
    ):
        self.variant = variant
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # 加载所有片段
        segments, labels, sample_ids = load_all_segments(variant, segment_length, sample_rate)
        
        if len(segments) == 0:
            raise ValueError(f"No segments loaded for variant '{variant}'")
        
        self.segments = segments
        self.sample_ids = sample_ids
        
        # 标签标准化
        self.scaler = StandardScaler()
        self.labels = self.scaler.fit_transform(labels.reshape(-1, 1)).flatten()
        
        # 按原始样本分组划分，确保同一样本的切片在同一split
        unique_samples = list(set(sample_ids))
        train_samples, test_samples = train_test_split(
            unique_samples, test_size=test_size, random_state=random_state
        )
        train_samples, val_samples = train_test_split(
            train_samples, test_size=val_size / (1 - test_size), random_state=random_state
        )
        
        train_samples_set = set(train_samples)
        val_samples_set = set(val_samples)
        test_samples_set = set(test_samples)
        
        self.train_indices = [i for i, sid in enumerate(sample_ids) if sid in train_samples_set]
        self.val_indices = [i for i, sid in enumerate(sample_ids) if sid in val_samples_set]
        self.test_indices = [i for i, sid in enumerate(sample_ids) if sid in test_samples_set]
    
    def _get_subset(self, indices: List[int]) -> Tuple[List[np.ndarray], np.ndarray]:
        segs = [self.segments[i] for i in indices]
        lbls = self.labels[indices]
        return segs, lbls
    
    def get_train_loader(self) -> DataLoader:
        segs, lbls = self._get_subset(self.train_indices)
        ds = FixedSegmentDataset(segs, lbls, is_training=True)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, 
                         num_workers=self.num_workers, pin_memory=True)
    
    def get_val_loader(self) -> DataLoader:
        segs, lbls = self._get_subset(self.val_indices)
        ds = FixedSegmentDataset(segs, lbls, is_training=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True)
    
    def get_test_loader(self) -> DataLoader:
        segs, lbls = self._get_subset(self.test_indices)
        ds = FixedSegmentDataset(segs, lbls, is_training=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True)
    
    def inverse_transform(self, labels: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
    
    def get_info(self) -> dict:
        return {
            'variant': self.variant,
            'total': len(self.segments),
            'train': len(self.train_indices),
            'val': len(self.val_indices),
            'test': len(self.test_indices),
        }


# ============================================================================
# Model Runners
# ============================================================================
def run_dalf(variant: str, seed: int, segment_length: float = 30.0) -> Dict[str, float]:
    """Run DALF model"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    dm = SegmentDataManager(variant=variant, segment_length=segment_length, 
                            batch_size=32, random_state=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()
    test_loader = dm.get_test_loader()

    model = DALFNet(DALFConfig(
        sample_rate=16000, num_filters=64, gabor_kernel=401,
        pool_kernel=401, pool_stride=160, meb_blocks=4,
        mssa_hidden=128, head_channels=128, dropout=0.2,
    )).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val, best_state = float("inf"), None
    for _ in range(50):
        train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, _, _ = dalf_evaluate(model, val_loader, device, criterion)
        scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    _, preds, labels = dalf_evaluate(model, test_loader, device, criterion)
    preds = dm.inverse_transform(preds)
    labels = dm.inverse_transform(labels)
    
    mae = np.mean(np.abs(labels - preds))
    rmse = np.sqrt(np.mean((labels - preds) ** 2))
    return {"MAE": mae, "RMSE": rmse}


def run_stfn(variant: str, seed: int, segment_length: float = 30.0) -> Dict[str, float]:
    """Run STFN model"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    dm = SegmentDataManager(variant=variant, segment_length=segment_length,
                            batch_size=20, random_state=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()
    test_loader = dm.get_test_loader()

    model = STFNModel(input_dim=1, dropout=0.5, prediction_steps=1)
    config = {"batch_size": 20, "max_epochs": 50, "learning_rate": 1e-3,
              "hcpc_weight": 0.1, "audio_length": segment_length}
    model = train_stfn_model(model, train_loader, val_loader, device, config)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for audios, lbls in test_loader:
            audios = audios.to(device)
            outputs, _ = model(audios)
            preds.extend(outputs.cpu().numpy().flatten())
            labels.extend(lbls.numpy().flatten())
    
    preds = dm.inverse_transform(np.array(preds))
    labels = dm.inverse_transform(np.array(labels))
    
    mae = np.mean(np.abs(labels - preds))
    rmse = np.sqrt(np.mean((labels - preds) ** 2))
    return {"MAE": mae, "RMSE": rmse}


class RawAudioDepAudioNet(nn.Module):
    """适配原始音频输入的DepAudioNet变体"""
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        # 使用1D CNN直接处理原始音频
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=16),  # 降采样
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )
        # 回归头
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, 1, samples)
        x = self.feature_extractor(x)  # (batch, 128, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 128)
        x, _ = self.lstm(x)  # (batch, seq_len, 256)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.fc(x)
        return x


def run_depaudionet(variant: str, seed: int, segment_length: float = 30.0) -> Dict[str, float]:
    """Run DepAudioNet model (原始音频版本)"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    dm = SegmentDataManager(variant=variant, segment_length=segment_length,
                            batch_size=16, random_state=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()
    test_loader = dm.get_test_loader()
    
    model = RawAudioDepAudioNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_val, best_state = float("inf"), None
    for epoch in range(80):
        model.train()
        for audios, lbls in train_loader:
            audios, lbls = audios.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(audios)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audios, lbls in val_loader:
                audios, lbls = audios.to(device), lbls.to(device)
                outputs = model(audios)
                val_loss += criterion(outputs, lbls).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for audios, lbls in test_loader:
            audios = audios.to(device)
            outputs = model(audios)
            preds.extend(outputs.cpu().numpy().flatten())
            labels.extend(lbls.numpy().flatten())
    
    preds = dm.inverse_transform(np.array(preds))
    labels = dm.inverse_transform(np.array(labels))
    
    mae = np.mean(np.abs(labels - preds))
    rmse = np.sqrt(np.mean((labels - preds) ** 2))
    return {"MAE": mae, "RMSE": rmse}


class RawAudioABAFNet(nn.Module):
    """适配原始音频输入的ABAFnet变体，使用Mel频谱特征"""
    def __init__(self, sample_rate: int = 16000, n_mels: int = 64, n_fft: int = 400, hop_length: int = 160):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Mel频谱2D CNN处理
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    
    def forward(self, x):
        # x: (batch, 1, samples)
        # 使用librosa计算Mel频谱（避免torchaudio版本问题）
        batch_size = x.shape[0]
        x_np = x.squeeze(1).cpu().numpy()  # (batch, samples)
        
        mel_list = []
        for i in range(batch_size):
            # 使用librosa提取mel频谱
            mel_spec = librosa.feature.melspectrogram(
                y=x_np[i],
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0
            )
            mel_spec = np.log(mel_spec + 1e-9)
            mel_list.append(mel_spec)
        
        mel = np.stack(mel_list, axis=0)  # (batch, n_mels, time)
        mel = torch.from_numpy(mel).float().to(x.device)
        mel = mel.unsqueeze(1)  # (batch, 1, n_mels, time)
        
        x = self.conv_layers(mel)  # (batch, 128, 4, 4)
        x = x.view(x.size(0), -1)  # (batch, 128*4*4)
        x = self.fc(x)
        return x


def run_abafnet(variant: str, seed: int, segment_length: float = 30.0) -> Dict[str, float]:
    """Run ABAFnet model (原始音频版本，自动提取Mel特征)"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    dm = SegmentDataManager(variant=variant, segment_length=segment_length,
                            batch_size=16, random_state=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()
    test_loader = dm.get_test_loader()
    
    model = RawAudioABAFNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_val, best_state = float("inf"), None
    for epoch in range(80):
        model.train()
        for audios, lbls in train_loader:
            audios, lbls = audios.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(audios)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audios, lbls in val_loader:
                audios, lbls = audios.to(device), lbls.to(device)
                outputs = model(audios)
                val_loss += criterion(outputs, lbls).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for audios, lbls in test_loader:
            audios = audios.to(device)
            outputs = model(audios)
            preds.extend(outputs.cpu().numpy().flatten())
            labels.extend(lbls.numpy().flatten())
    
    preds = dm.inverse_transform(np.array(preds))
    labels = dm.inverse_transform(np.array(labels))
    
    mae = np.mean(np.abs(labels - preds))
    rmse = np.sqrt(np.mean((labels - preds) ** 2))
    return {"MAE": mae, "RMSE": rmse}


MODEL_RUNNERS: Dict[str, Callable] = {
    "DepAudioNet": run_depaudionet,
    "ABAFnet": run_abafnet,
    "DALF": run_dalf,
    "STFN": run_stfn,
}


# ============================================================================
# Main
# ============================================================================
def main():
    variants = ["pre", "cue-exclude"]
    seeds = list(range(5))
    segment_length = 30.0
    
    print("\n" + "=" * 70)
    print("E-DAIC Fixed-Length Segment Experiments")
    print(f"Segment length: {segment_length}s | Seeds: {len(seeds)} | Variants: {variants}")
    print("=" * 70)
    
    # 预加载数据信息
    for variant in variants:
        try:
            dm = SegmentDataManager(variant=variant, segment_length=segment_length, random_state=0)
            info = dm.get_info()
            print(f"\n[{variant}] Total: {info['total']} segments | "
                  f"Train: {info['train']} | Val: {info['val']} | Test: {info['test']}")
        except Exception as e:
            print(f"\n[{variant}] Error loading data: {e}")
    
    results_table: List[Dict] = []

    for model_name, runner in MODEL_RUNNERS.items():
        for variant in variants:
            mae_list, rmse_list = [], []
            
            for seed in seeds:
                print(f"\rRunning {model_name} | {variant} | seed {seed}...", end="", flush=True)
                try:
                    metrics = runner(variant, seed, segment_length)
                    mae_list.append(metrics["MAE"])
                    rmse_list.append(metrics["RMSE"])
                except Exception as e:
                    print(f"\n  Error: {e}")
                    continue
            
            if mae_list:
                mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
                rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
                results_table.append({
                    "Model": model_name,
                    "Variant": variant,
                    "MAE": f"{mae_mean:.3f}±{mae_std:.3f}",
                    "RMSE": f"{rmse_mean:.3f}±{rmse_std:.3f}",
                })
                print(f"\r{model_name} | {variant}: MAE={mae_mean:.3f}±{mae_std:.3f}, "
                      f"RMSE={rmse_mean:.3f}±{rmse_std:.3f}")

    # 打印结果表
    print("\n" + "=" * 70)
    print("Results (5 seeds, 7:2:1 split, 30s fixed segments)")
    print("=" * 70)
    print(f"{'Model':<14} {'Variant':<14} {'MAE':<18} {'RMSE':<18}")
    print("-" * 70)
    for row in results_table:
        print(f"{row['Model']:<14} {row['Variant']:<14} {row['MAE']:<18} {row['RMSE']:<18}")
    
    # 保存结果
    results_df = pd.DataFrame(results_table)
    results_path = PROJECT_ROOT / "CueGate" / "Baseline" / "experiment_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
