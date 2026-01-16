#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STFN数据加载器
用于加载和处理音频文件及对应的量表总分数据
"""

import os
import torch
# import torchaudio  # 未使用，移除以避免版本冲突
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
import librosa
import warnings
warnings.filterwarnings('ignore')


class AudioDataset(Dataset):
    """音频数据集类"""
    
    def __init__(
        self, 
        audio_paths: List[str],
        labels: np.ndarray,
        sample_rate: int = 16000,
        segment_length: float = 3.0,
        is_training: bool = True
    ):
        """
        初始化音频数据集
        
        Args:
            audio_paths: 音频文件路径列表
            labels: 对应的标签（量表总分）
            sample_rate: 采样率
            segment_length: 音频片段长度（秒）
            is_training: 是否为训练模式
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)
        self.is_training = is_training
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # 加载音频
        try:
            # 使用librosa加载音频，确保采样率一致
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            audio = torch.FloatTensor(audio)
        except Exception as e:
            print(f"加载音频文件失败: {audio_path}, 错误: {e}")
            # 返回零音频作为备选
            audio = torch.zeros(self.segment_samples)
        
        # 音频预处理
        audio = self._preprocess_audio(audio)
        
        return audio, torch.FloatTensor([label])
    
    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """音频预处理"""
        # 归一化
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()
        
        # 分段处理
        if len(audio) > self.segment_samples:
            if self.is_training:
                # 训练时随机选择片段
                start_idx = torch.randint(0, len(audio) - self.segment_samples + 1, (1,)).item()
                audio = audio[start_idx:start_idx + self.segment_samples]
            else:
                # 测试时选择中间片段
                start_idx = (len(audio) - self.segment_samples) // 2
                audio = audio[start_idx:start_idx + self.segment_samples]
        elif len(audio) < self.segment_samples:
            # 如果音频太短，进行零填充
            padding = self.segment_samples - len(audio)
            audio = torch.cat([audio, torch.zeros(padding)])
        
        return audio


class STFNDataLoader:
    """STFN数据加载器"""
    
    def __init__(
        self,
        audio_dir: str,
        questionnaire_path: str,
        target_task: str = 'total_score',
        segment_level: str = "primary",
        audio_variant: str = "pre",
        sample_rate: int = 16000,
        segment_length: float = 3.0,
        batch_size: int = 16,
        num_workers: int = 4,
        test_size: float = 0.1,
        val_size: float = 0.2,
        random_state: int = 42
    ):
        
        self.audio_dir = audio_dir
        self.questionnaire_path = questionnaire_path
        self.target_task = target_task
        self.segment_level = segment_level
        self.audio_variant = audio_variant
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # 标签标准化器
        self.label_scaler = StandardScaler()
        
        # 加载和处理数据
        self._load_data()
        self._split_data()
    
    def _load_data(self):
        print("正在加载数据...")
        
        # 加载问卷数据
        questionnaire_df = pd.read_csv(self.questionnaire_path)
        
        # DAIC数据集：使用PHQ_8Total作为标签
        label_column = 'PHQ_8Total'
        
        # 获取音频文件列表
        audio_files = []
        labels = []
        
        for _, row in questionnaire_df.iterrows():
            audio_id = str(int(row['Participant_ID']))

            participant_root = os.path.join(self.audio_dir, f"{audio_id}_P", self.segment_level)
            audio_path = os.path.join(participant_root, f"{self.audio_variant}.wav")

            if os.path.exists(audio_path):
                audio_files.append(audio_path)
                labels.append(row[label_column])
            else:
                print(f"警告: 音频文件不存在: {audio_path}")
        
        print(f"成功加载 {len(audio_files)} 个音频文件，目标任务: {getattr(self, 'target_task', 'total_score')}")
        
        self.audio_paths = audio_files
        self.labels = np.array(labels)
        
        # 标准化标签
        self.labels_normalized = self.label_scaler.fit_transform(self.labels.reshape(-1, 1)).flatten()
    
    def _split_data(self):
        """数据分割"""
        print("正在分割数据集...")
        
        # 首先分割出测试集
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.audio_paths, self.labels_normalized,
            test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        
        # 再从训练集中分割出验证集
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.val_size / (1 - self.test_size), random_state=self.random_state, stratify=None
        )
        
        # 保存分割结果
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        
        print(f"训练集: {len(train_paths)} 样本")
        print(f"验证集: {len(val_paths)} 样本")
        print(f"测试集: {len(test_paths)} 样本")
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
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
        """获取验证数据加载器"""
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
        """获取测试数据加载器"""
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
        """将标准化的标签转换回原始标签"""
        return self.label_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
    
    def get_data_info(self) -> Dict:
        """获取数据集信息"""
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


def collate_fn(batch):
    """自定义批次整理函数"""
    audios, labels = zip(*batch)
    
    # 将音频堆叠为批次
    audios = torch.stack(audios, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return audios, labels


if __name__ == "__main__":
    # 测试数据加载器
    audio_dir = "/home/a001/xuxiao/LIRA/dataset/audio"
    questionnaire_path = "/home/a001/xuxiao/LIRA/dataset/raw_info.csv"
    
    # 创建数据加载器
    data_loader = STFNDataLoader(
        audio_dir=audio_dir,
        questionnaire_path=questionnaire_path,
        batch_size=8,
        num_workers=2
    )
    
    # 打印数据信息
    print("数据集信息:")
    for key, value in data_loader.get_data_info().items():
        print(f"  {key}: {value}")
    
    # 测试训练数据加载器
    train_loader = data_loader.get_train_loader()
    print(f"\n训练批次数: {len(train_loader)}")
    
    # 测试一个批次
    for batch_idx, (audios, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx}: 音频形状 {audios.shape}, 标签形状 {labels.shape}")
        print(f"音频范围: [{audios.min().item():.4f}, {audios.max().item():.4f}]")
        print(f"标签范围: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
        break 
