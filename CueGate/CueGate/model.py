"""
CueGate Model: 稀疏语义门控网络用于Cue检测

核心设计：
1. 双流架构：Acoustic Stream (局部) + Semantic Stream (全局上下文)
2. 稀疏门控：显式建模cue的稀疏性
3. 对比学习：帧级对比损失处理极端不平衡

输入：原始波形 (batch, samples)
输出：帧级cue概率 (batch, num_frames)
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CueGateConfig:
    """CueGate模型配置"""
    sample_rate: int = 16000
    # Acoustic Stream - 增大通道数
    sinc_filters: int = 80
    sinc_kernel: int = 251
    acoustic_channels: int = 128
    acoustic_layers: int = 4
    # Semantic Stream - 增大通道数
    semantic_channels: int = 128
    semantic_layers: int = 4
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16)  # 更细粒度的多尺度膨胀率
    # Sparse Gate
    gate_dim: int = 128
    sparsity_ratio: float = 0.1  # 期望cue占比
    # Frame settings
    frame_stride: int = 160  # 10ms @ 16kHz
    # General
    dropout: float = 0.15
    # Focal Loss参数
    focal_alpha: float = 0.8  # 正样本权重（增大对正样本的关注）
    focal_gamma: float = 2.5  # 聚焦参数（增大对困难样本的关注）


class SincConv(nn.Module):
    """
    可学习的Sinc滤波器组
    比固定Mel滤波器更灵活，能自适应学习语音特征
    """
    
    def __init__(
        self,
        out_channels: int = 64,
        kernel_size: int = 251,
        sample_rate: int = 16000,
        min_freq: float = 50.0,
        max_freq: float = 8000.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # 初始化频率参数（可学习）
        low_hz = min_freq
        high_hz = max_freq
        
        # Mel scale initialization
        mel_low = 2595 * math.log10(1 + low_hz / 700)
        mel_high = 2595 * math.log10(1 + high_hz / 700)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        self.low_hz_ = nn.Parameter(hz_points[:-1].clone())
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).clone())
        
        # Hamming window
        n = torch.arange(kernel_size).float()
        self.register_buffer('window', 0.54 - 0.46 * torch.cos(2 * math.pi * n / kernel_size))
        self.register_buffer('n_', (kernel_size - 1) / 2 - n)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, samples)
        Returns:
            (batch, out_channels, frames)
        """
        # 计算滤波器
        low = self.low_hz_.abs() + 1
        high = (low + self.band_hz_.abs()).clamp(max=self.sample_rate / 2)
        
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate
        
        # Sinc滤波器
        f_low = f_low.unsqueeze(1)  # (out_channels, 1)
        f_high = f_high.unsqueeze(1)
        n = self.n_.unsqueeze(0)  # (1, kernel_size)
        
        # 避免除零
        n_safe = torch.where(n == 0, torch.ones_like(n), n)
        
        low_pass = 2 * f_low * torch.sin(2 * math.pi * f_low * n_safe) / (2 * math.pi * f_low * n_safe)
        high_pass = 2 * f_high * torch.sin(2 * math.pi * f_high * n_safe) / (2 * math.pi * f_high * n_safe)
        
        # 处理n=0的情况
        low_pass = torch.where(n == 0, 2 * f_low, low_pass)
        high_pass = torch.where(n == 0, 2 * f_high, high_pass)
        
        band_pass = (high_pass - low_pass) * self.window
        band_pass = band_pass / band_pass.abs().sum(dim=1, keepdim=True)
        
        filters = band_pass.unsqueeze(1)  # (out_channels, 1, kernel_size)
        
        return F.conv1d(x, filters, padding=self.kernel_size // 2)


class TemporalBlock(nn.Module):
    """时序卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.gelu(self.norm2(self.conv2(x)))
        x = self.dropout(x)
        return x + res


class AcousticStream(nn.Module):
    """
    声学流：捕捉局部声学特征
    使用SincConv + 时序卷积
    """
    
    def __init__(self, config: CueGateConfig):
        super().__init__()
        self.sinc = SincConv(
            out_channels=config.sinc_filters,
            kernel_size=config.sinc_kernel,
            sample_rate=config.sample_rate,
        )
        
        # 下采样到帧级别
        self.downsample = nn.Conv1d(
            config.sinc_filters, config.acoustic_channels,
            kernel_size=config.frame_stride, stride=config.frame_stride
        )
        
        # 时序卷积栈
        self.blocks = nn.ModuleList([
            TemporalBlock(config.acoustic_channels, config.acoustic_channels, 
                         dropout=config.dropout)
            for _ in range(config.acoustic_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, samples)
        Returns:
            (batch, channels, frames)
        """
        x = self.sinc(x)
        x = torch.abs(x)  # 取幅度
        x = self.downsample(x)
        
        for block in self.blocks:
            x = block(x)
        
        return x


class SemanticStream(nn.Module):
    """
    语义流：捕捉多尺度上下文
    使用不同膨胀率的卷积捕捉词/短语/句子级别的上下文
    """
    
    def __init__(self, config: CueGateConfig):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Conv1d(1, config.semantic_channels, 
                                    kernel_size=config.frame_stride, 
                                    stride=config.frame_stride)
        
        # 多尺度膨胀卷积
        self.multi_scale = nn.ModuleList()
        for dilation in config.dilations:
            self.multi_scale.append(
                nn.Sequential(
                    nn.Conv1d(config.semantic_channels, config.semantic_channels,
                             kernel_size=7, padding=3 * dilation, dilation=dilation),
                    nn.GroupNorm(8, config.semantic_channels),
                    nn.GELU(),
                )
            )
        
        # 融合多尺度
        self.fuse = nn.Conv1d(
            config.semantic_channels * len(config.dilations),
            config.semantic_channels, 1
        )
        
        # 额外的时序建模
        self.blocks = nn.ModuleList([
            TemporalBlock(config.semantic_channels, config.semantic_channels,
                         kernel_size=7, dilation=2**i, dropout=config.dropout)
            for i in range(config.semantic_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, samples)
        Returns:
            (batch, channels, frames)
        """
        x = self.input_proj(x)
        
        # 多尺度卷积
        multi = [conv(x) for conv in self.multi_scale]
        x = self.fuse(torch.cat(multi, dim=1))
        
        for block in self.blocks:
            x = block(x)
        
        return x


class SparseGate(nn.Module):
    """
    稀疏门控模块
    
    核心思想：
    - 融合声学和语义特征
    - 输出稀疏的门控值（大部分接近0，少量接近1）
    - 使用Gumbel-Softmax或Top-K实现可微分稀疏选择
    """
    
    def __init__(self, config: CueGateConfig):
        super().__init__()
        in_dim = config.acoustic_channels + config.semantic_channels
        
        self.fuse = nn.Sequential(
            nn.Conv1d(in_dim, config.gate_dim, 1),
            nn.GroupNorm(8, config.gate_dim),
            nn.GELU(),
            nn.Conv1d(config.gate_dim, config.gate_dim, 7, padding=3),
            nn.GroupNorm(8, config.gate_dim),
            nn.GELU(),
        )
        
        self.gate_proj = nn.Conv1d(config.gate_dim, 1, 1)
        self.sparsity_ratio = config.sparsity_ratio
    
    def forward(self, acoustic: torch.Tensor, semantic: torch.Tensor, 
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            acoustic: (batch, acoustic_channels, frames)
            semantic: (batch, semantic_channels, frames)
            training: 是否训练模式
        
        Returns:
            gate: (batch, frames) 门控值
            gate_logits: (batch, frames) 用于损失计算
        """
        # 融合特征
        fused = torch.cat([acoustic, semantic], dim=1)
        fused = self.fuse(fused)
        
        gate_logits = self.gate_proj(fused).squeeze(1)  # (batch, frames)
        
        if training:
            # 训练时使用Gumbel-Sigmoid实现可微分采样
            gate = self._gumbel_sigmoid(gate_logits, temperature=0.5)
        else:
            # 推理时使用阈值或Top-K
            gate = torch.sigmoid(gate_logits)
        
        return gate, gate_logits
    
    def _gumbel_sigmoid(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Gumbel-Sigmoid: 可微分的二值采样"""
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            return torch.sigmoid((logits + gumbel_noise) / temperature)
        return torch.sigmoid(logits)


class ContrastiveHead(nn.Module):
    """
    对比学习头：输出用于对比损失的特征
    """
    
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, frames, in_dim)
        Returns:
            (batch, frames, proj_dim) L2归一化的特征
        """
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class CueGate(nn.Module):
    """
    CueGate: 稀疏语义门控网络用于Cue检测
    
    使用方式：
        model = CueGate()
        # 训练
        outputs = model(waveform, labels)  # labels: (batch, frames) 0/1
        loss = outputs['loss']
        
        # 推理
        outputs = model(waveform)
        cue_probs = outputs['cue_probs']  # (batch, frames)
    """
    
    def __init__(self, config: Optional[CueGateConfig] = None):
        super().__init__()
        self.config = config or CueGateConfig()
        
        # 双流架构
        self.acoustic_stream = AcousticStream(self.config)
        self.semantic_stream = SemanticStream(self.config)
        
        # 稀疏门控
        self.sparse_gate = SparseGate(self.config)
        
        # 分类头
        feat_dim = self.config.acoustic_channels + self.config.semantic_channels
        self.classifier = nn.Sequential(
            nn.Conv1d(feat_dim, self.config.gate_dim, 1),
            nn.GroupNorm(8, self.config.gate_dim),
            nn.GELU(),
            nn.Conv1d(self.config.gate_dim, 1, 1),
        )
        
        # 对比学习头
        self.contrastive_head = ContrastiveHead(feat_dim, proj_dim=64)
        
        # 帧设置
        self.frame_stride = self.config.frame_stride
        self.sample_rate = self.config.sample_rate
    
    def forward(
        self,
        waveform: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            waveform: (batch, samples) 或 (batch, 1, samples)
            labels: (batch, frames) 帧级标签 0/1，训练时需要
        
        Returns:
            dict containing:
                - cue_probs: (batch, frames) 帧级cue概率
                - gate: (batch, frames) 门控值
                - loss: 总损失（训练时）
                - cls_loss: 分类损失
                - contrast_loss: 对比损失
                - sparsity_loss: 稀疏正则
        """
        # 确保输入形状
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (batch, 1, samples)
        
        # 双流特征提取
        acoustic_feat = self.acoustic_stream(waveform)  # (batch, C1, frames)
        semantic_feat = self.semantic_stream(waveform)  # (batch, C2, frames)
        
        # 确保帧数一致
        min_frames = min(acoustic_feat.size(2), semantic_feat.size(2))
        acoustic_feat = acoustic_feat[:, :, :min_frames]
        semantic_feat = semantic_feat[:, :, :min_frames]
        
        # 稀疏门控
        gate, gate_logits = self.sparse_gate(
            acoustic_feat, semantic_feat, training=self.training
        )
        
        # 融合特征
        fused = torch.cat([acoustic_feat, semantic_feat], dim=1)  # (batch, C1+C2, frames)
        
        # 分类
        cls_logits = self.classifier(fused).squeeze(1)  # (batch, frames)
        # 使用加性门控而不是乘性，避免概率被压制过低
        # cue_probs = sigmoid(cls_logits + gate_logits)，这样门控起到偏置作用
        cue_probs = torch.sigmoid(cls_logits + gate_logits)  # 加性门控
        
        outputs = {
            'cue_probs': cue_probs,
            'gate': gate,
            'gate_logits': gate_logits,
            'cls_logits': cls_logits,
            'combined_logits': cls_logits + gate_logits,  # 用于损失计算
            'num_frames': min_frames,
        }
        
        # 计算损失（训练时）
        if labels is not None:
            # 对齐标签和预测的帧数
            if labels.size(1) != min_frames:
                labels = F.interpolate(
                    labels.unsqueeze(1).float(), 
                    size=min_frames, 
                    mode='nearest'
                ).squeeze(1)
            
            # 1. 分类损失（Focal Loss处理不平衡）- 使用组合logits
            combined_logits = cls_logits + gate_logits
            cls_loss = self._focal_loss(combined_logits, labels)
            
            # 2. 对比损失
            fused_t = fused.permute(0, 2, 1)  # (batch, frames, channels)
            contrast_feat = self.contrastive_head(fused_t)
            contrast_loss = self._contrastive_loss(contrast_feat, labels)
            
            # 3. 稀疏正则
            sparsity_loss = self._sparsity_loss(gate, labels)
            
            # 4. 门控引导损失
            gate_loss = F.binary_cross_entropy_with_logits(gate_logits, labels)
            
            # 总损失
            loss = cls_loss + 0.5 * contrast_loss + 0.1 * sparsity_loss + 0.5 * gate_loss
            
            outputs.update({
                'loss': loss,
                'cls_loss': cls_loss,
                'contrast_loss': contrast_loss,
                'sparsity_loss': sparsity_loss,
                'gate_loss': gate_loss,
            })
        
        return outputs
    
    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal Loss: 处理类别不平衡，使用config中的参数"""
        alpha = self.config.focal_alpha
        gamma = self.config.focal_gamma
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = alpha * labels + (1 - alpha) * (1 - labels)
        focal_loss = focal_weight * ((1 - pt) ** gamma) * bce
        return focal_loss.mean()
    
    def _contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor,
                          temperature: float = 0.1) -> torch.Tensor:
        """
        帧级对比损失
        拉近cue帧，推远non-cue帧
        """
        batch_size, num_frames, dim = features.shape
        
        # 展平
        features = features.reshape(-1, dim)  # (batch*frames, dim)
        labels = labels.reshape(-1)  # (batch*frames,)
        
        cue_mask = labels > 0.5
        non_cue_mask = labels < 0.5
        
        if cue_mask.sum() < 2 or non_cue_mask.sum() < 2:
            return torch.tensor(0.0, device=features.device)
        
        cue_features = features[cue_mask]
        non_cue_features = features[non_cue_mask]
        
        # 计算cue特征的中心
        cue_center = cue_features.mean(dim=0, keepdim=True)
        
        # Cue帧应该靠近中心
        cue_sim = F.cosine_similarity(cue_features, cue_center.expand_as(cue_features))
        cue_loss = (1 - cue_sim).mean()
        
        # Non-cue帧应该远离中心
        # 采样一部分non-cue帧避免计算量过大
        if non_cue_features.size(0) > 100:
            idx = torch.randperm(non_cue_features.size(0))[:100]
            non_cue_features = non_cue_features[idx]
        
        non_cue_sim = F.cosine_similarity(non_cue_features, cue_center.expand_as(non_cue_features))
        non_cue_loss = F.relu(non_cue_sim - 0.2).mean()  # margin = 0.2
        
        return cue_loss + non_cue_loss
    
    def _sparsity_loss(self, gate: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """稀疏正则：鼓励门控稀疏"""
        # 实际稀疏度
        actual_sparsity = gate.mean()
        # 目标稀疏度（根据标签）
        target_sparsity = labels.mean()
        
        # L1正则
        sparsity_reg = torch.abs(actual_sparsity - target_sparsity)
        
        return sparsity_reg
    
    def get_num_frames(self, num_samples: int) -> int:
        """计算给定样本数对应的帧数"""
        return num_samples // self.frame_stride
    
    def frames_to_time(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """将帧索引转换为时间（秒）"""
        return frame_indices.float() * self.frame_stride / self.sample_rate
    
    @torch.no_grad()
    def detect(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.5,
        min_duration: float = 0.1,
        merge_gap: float = 0.05,
    ) -> list:
        """
        检测cue片段
        
        Args:
            waveform: (samples,) 或 (1, samples) 单个音频
            threshold: cue概率阈值
            min_duration: 最小cue时长（秒）
            merge_gap: 合并间隔小于此值的片段
        
        Returns:
            List of dicts: [{'start': float, 'end': float, 'score': float}, ...]
        """
        self.eval()
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # (1, 1, samples)
        
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        
        outputs = self(waveform)
        probs = outputs['cue_probs'][0].cpu().numpy()  # (frames,)
        
        # 二值化
        binary = probs > threshold
        
        # 找连续区域
        spans = []
        in_span = False
        start_frame = 0
        
        for i, is_cue in enumerate(binary):
            if is_cue and not in_span:
                in_span = True
                start_frame = i
            elif not is_cue and in_span:
                in_span = False
                spans.append((start_frame, i))
        
        if in_span:
            spans.append((start_frame, len(binary)))
        
        # 转换为时间并过滤
        frame_duration = self.frame_stride / self.sample_rate
        min_frames = int(min_duration / frame_duration)
        merge_frames = int(merge_gap / frame_duration)
        
        # 过滤太短的片段
        spans = [(s, e) for s, e in spans if e - s >= min_frames]
        
        # 合并相近的片段
        if spans:
            merged = [spans[0]]
            for start, end in spans[1:]:
                if start - merged[-1][1] <= merge_frames:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            spans = merged
        
        # 转换为结果格式
        results = []
        for start_frame, end_frame in spans:
            start_time = start_frame * frame_duration
            end_time = end_frame * frame_duration
            score = probs[start_frame:end_frame].mean()
            results.append({
                'start': round(start_time, 3),
                'end': round(end_time, 3),
                'score': round(float(score), 3),
            })
        
        return results


def create_model(config: Optional[CueGateConfig] = None) -> CueGate:
    """创建CueGate模型"""
    return CueGate(config)


if __name__ == "__main__":
    # 测试模型
    config = CueGateConfig()
    model = CueGate(config)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 4
    duration = 5.0  # 5秒
    samples = int(duration * config.sample_rate)
    
    waveform = torch.randn(batch_size, samples)
    num_frames = model.get_num_frames(samples)
    labels = torch.zeros(batch_size, num_frames)
    labels[:, 10:15] = 1  # 模拟一些cue帧
    
    outputs = model(waveform, labels)
    print(f"\nInput shape: {waveform.shape}")
    print(f"Output cue_probs shape: {outputs['cue_probs'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # 测试检测
    model.eval()
    test_audio = torch.randn(samples)
    detections = model.detect(test_audio, threshold=0.3)
    print(f"\nDetections: {detections}")

