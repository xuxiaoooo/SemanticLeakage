import torch
import torch.nn as nn

class SpectrogramModel(nn.Module):
    """Spectrogram特征提取模型"""
    def __init__(self, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        # 输入: (batch, 1, freq_bins, time_frames)
        # 使用2D卷积处理频谱图
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        # x shape: (batch, 1, freq_bins, time_frames)
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.global_pool(x)  # (batch, 64, 1, 1)
        x = torch.flatten(x, start_dim=1)  # (batch, 64)
        x = self.fc(x)  # (batch, 128)
        return x




