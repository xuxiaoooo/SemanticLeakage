
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import torchaudio  # 未使用，移除以避免版本冲突
import librosa
import warnings

class DepAudioNetDataLoader:
    
    def __init__(
        self,
        audio_dir,
        questionnaire_path,
        target_task='total_score',
        segment_level: str = "primary",
        audio_variant: str = "pre",
        batch_size=32,
        num_workers=4,
        segment_length=3.0,
        random_state=42,
        val_size=0.2,
        test_size=0.1
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
        
        # Audio sample rate
        self.sample_rate = 16000
        
        # Label scaler
        self.label_scaler = StandardScaler()
        
        # Load data
        self._load_data()
        
        # Split data
        self._split_data()
    
    def _load_data(self):
        """Load data"""
        print("Loading data...")
        
        # Load questionnaire data
        questionnaire_df = pd.read_csv(self.questionnaire_path)
        
        # DAIC数据集：使用PHQ_8Total作为标签
        label_column = 'PHQ_8Total'
        
        # Get audio file list
        audio_files = []
        labels = []

        for _, row in questionnaire_df.iterrows():
            # DAIC数据格式：第一列是Participant_ID
            audio_id = str(int(row['Participant_ID']))

            base_dir = os.path.join(self.audio_dir, f"{audio_id}_P", self.segment_level)
            audio_path = os.path.join(base_dir, f"{self.audio_variant}.wav")

            if os.path.exists(audio_path):
                audio_files.append(audio_path)
                labels.append(row[label_column])
            else:
                print(f"Warning: Audio file not found for participant {audio_id}: {audio_path}")
        
        print(f"Successfully loaded {len(audio_files)} audio files, target task: {self.target_task}")
        
        self.audio_paths = audio_files
        self.labels = np.array(labels)
        self.labels_normalized = self.label_scaler.fit_transform(self.labels.reshape(-1, 1)).flatten()
    
    def _split_data(self):
        print("Splitting dataset...")
        
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.audio_paths, self.labels_normalized,
            test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.val_size / (1 - self.test_size), 
            random_state=self.random_state, stratify=None
        )
        
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        
        print(f"Training set: {len(train_paths)} samples")
        print(f"Validation set: {len(val_paths)} samples")
        print(f"Test set: {len(test_paths)} samples")
    
    def get_train_loader(self) -> DataLoader:
        train_dataset = AudioDataset(
            self.train_paths, self.train_labels,
            self.sample_rate, self.segment_length, is_training=True
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_loader(self) -> DataLoader:
        val_dataset = AudioDataset(
            self.val_paths, self.val_labels,
            self.sample_rate, self.segment_length, is_training=False
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self) -> DataLoader:
        test_dataset = AudioDataset(
            self.test_paths, self.test_labels,
            self.sample_rate, self.segment_length, is_training=False
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def inverse_transform_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.label_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
    
    def get_data_info(self):
        return {
            'train_size': len(self.train_paths),
            'val_size': len(self.val_paths),
            'test_size': len(self.test_paths),
            'total_size': len(self.audio_paths),
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'label_mean': float(self.labels.mean()),
            'label_std': float(self.labels.std()),
            'label_min': float(self.labels.min()),
            'label_max': float(self.labels.max())
        }


class AudioDataset(Dataset):
    
    def __init__(
        self, 
        audio_paths, 
        labels, 
        sample_rate=16000, 
        segment_length=3.0, 
        is_training=True
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.is_training = is_training
        self.segment_samples = int(segment_length * sample_rate)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 使用librosa加载音频，避免torchcodec依赖问题
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                waveform = torch.FloatTensor(audio).unsqueeze(0)  # shape: (1, length)
                
                if waveform.shape[1] < self.segment_samples:
                    padding = self.segment_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding), "constant", 0)
                else:
                    if self.is_training:
                        max_start = waveform.shape[1] - self.segment_samples
                        start = torch.randint(0, max_start + 1, (1,)).item()
                    else:
                        start = max(0, (waveform.shape[1] - self.segment_samples) // 2)
                    
                    waveform = waveform[:, start:start + self.segment_samples]
                
                if self.is_training:
                    if torch.rand(1).item() < 0.5:
                        volume_factor = 0.5 + torch.rand(1).item()
                        waveform = waveform * volume_factor
                    
                    if torch.rand(1).item() < 0.3:
                        noise_level = 0.005 * torch.rand(1).item()
                        noise = torch.randn_like(waveform) * noise_level
                        waveform = waveform + noise
                
                features = waveform.squeeze(0)
                # 如果特征长度超过segment_samples，进行池化
                target_length = 100  # 目标长度
                if features.shape[0] > target_length:
                    pool = torch.nn.AvgPool1d(kernel_size=features.shape[0] // target_length, stride=features.shape[0] // target_length)
                    features = pool(features.unsqueeze(0)).squeeze(0)
                elif features.shape[0] < target_length:
                    # 如果太短，进行填充
                    padding = target_length - features.shape[0]
                    features = torch.nn.functional.pad(features, (0, padding), "constant", 0)
                
                features = features.unsqueeze(1)
                
                return features, label
                
        except Exception as e:
            print(f"Error loading audio file: {audio_path}, error: {str(e)}")
            return torch.zeros((100, 1)), label

if __name__ == "__main__":
    data_loader = DepAudioNetDataLoader(
        audio_dir="/home/a001/xuxiao/LIRA/dataset/audio",
        questionnaire_path="/home/a001/xuxiao/LIRA/dataset/raw_info.csv",
        target_task="total_score",
        batch_size=32
    )
    
    train_loader = data_loader.get_train_loader()
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: features shape: {features.shape}, labels shape: {labels.shape}")
        if batch_idx >= 2:
            break
