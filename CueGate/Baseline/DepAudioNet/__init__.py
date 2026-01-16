"""
DepAudioNet模块 - 基于音频的抑郁症检测
参考: Ma et al. (2016) DepAudioNet: An Efficient Deep Model for Audio Based Depression Classification
"""

from .depaudionet_model import DepAudioNet, DepAudioNetConfig
from .data_loader import DepAudioNetDataLoader
from .train import train_depaudionet

__all__ = ['DepAudioNet', 'DepAudioNetConfig', 'DepAudioNetDataLoader', 'train_depaudionet'] 