import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import librosa


class DALFDataLoader:
    """DALF 数据加载器：读取wav -> 6s段 -> (B,1,T) 波形张量"""

    def __init__(
        self,
        audio_dir: str,
        questionnaire_path: str,
        target_task: str = 'total_score',
        segment_level: str = "primary",
        audio_variant: str = "pre",
        batch_size: int = 32,
        num_workers: int = 4,
        segment_length: float = 6.0,
        random_state: int = 42,
        val_size: float = 0.2,
        test_size: float = 0.1,
        sample_rate: int = 16000
    ):
        self.audio_dir = audio_dir
        self.questionnaire_path = questionnaire_path
        self.target_task = target_task
        self.segment_level = segment_level
        self.audio_variant = audio_variant
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_length = segment_length
        self.random_state = random_state
        self.val_size = val_size
        self.test_size = test_size
        self.sample_rate = sample_rate

        self.label_scaler = StandardScaler()

        self._load_data()
        self._split_data()

    def _load_data(self) -> None:
        print("Loading data...")
        df = pd.read_csv(self.questionnaire_path)
        label_column = 'PHQ_8Total'

        audio_files, labels = [], []
        for _, row in df.iterrows():
            pid = str(int(row['Participant_ID']))

            participant_root = os.path.join(self.audio_dir, f"{pid}_P", self.segment_level)
            audio_path = os.path.join(participant_root, f"{self.audio_variant}.wav")

            if audio_path and os.path.exists(audio_path):
                audio_files.append(audio_path)
                labels.append(row[label_column])
            else:
                print(f"Warning: Audio file not found: {audio_path}")

        self.audio_paths = audio_files
        self.labels = np.array(labels, dtype=np.float32)
        self.labels_norm = self.label_scaler.fit_transform(self.labels.reshape(-1, 1)).flatten()
        print(f"Loaded {len(self.audio_paths)} files. Sample rate={self.sample_rate}, segment={self.segment_length}s")

    def _split_data(self) -> None:
        print("Splitting dataset...")
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.audio_paths, self.labels_norm, test_size=self.test_size, random_state=self.random_state
        )
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.val_size / (1.0 - self.test_size), random_state=self.random_state
        )
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        print(f"Train/Val/Test: {len(train_paths)}/{len(val_paths)}/{len(test_paths)}")

    def get_train_loader(self) -> DataLoader:
        ds = _WaveformDataset(self.train_paths, self.train_labels, self.sample_rate, self.segment_length, True)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def get_val_loader(self) -> DataLoader:
        ds = _WaveformDataset(self.val_paths, self.val_labels, self.sample_rate, self.segment_length, False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def get_test_loader(self) -> DataLoader:
        ds = _WaveformDataset(self.test_paths, self.test_labels, self.sample_rate, self.segment_length, False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def inverse_transform_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.label_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()

    def get_data_info(self) -> Dict:
        return {
            'train_size': len(self.train_paths),
            'val_size': len(self.val_paths),
            'test_size': len(self.test_paths),
            'total_size': len(self.audio_paths),
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'label_min': float(self.labels.min()),
            'label_max': float(self.labels.max())
        }


class _WaveformDataset(Dataset):
    def __init__(self, paths, labels, sample_rate: int, segment_length: float, is_training: bool):
        self.paths = paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.segment_len_samples = int(segment_length * sample_rate)
        self.is_training = is_training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
                x = torch.from_numpy(wav).float()  # [T]

            if x.numel() < self.segment_len_samples:
                pad = self.segment_len_samples - x.numel()
                x = F.pad(x, (0, pad))
            else:
                if self.is_training:
                    max_start = max(0, x.numel() - self.segment_len_samples)
                    start = int(torch.randint(0, max_start + 1, (1,)).item())
                else:
                    start = max(0, (x.numel() - self.segment_len_samples) // 2)
                x = x[start:start + self.segment_len_samples]

            # 轻微增广（仅训练）
            if self.is_training:
                if torch.rand(1).item() < 0.5:
                    x = x * (0.6 + 0.8 * torch.rand(1).item())
                if torch.rand(1).item() < 0.3:
                    noise = 0.003 * torch.randn_like(x)
                    x = x + noise

            x = x.unsqueeze(0)  # [1, T]
            return x, float(label)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return torch.zeros(1, self.segment_len_samples), float(label)
