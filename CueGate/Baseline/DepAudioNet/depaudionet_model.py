

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DepAudioNetConfig:
    # 音频特征配置
    input_features: int = 74 
    sequence_length: int = 100
    
    # 1D CNN配置
    cnn_channels: list = None 
    cnn_kernel_sizes: list = None
    cnn_stride: int = 1
    cnn_padding: int = 1
    
    # LSTM配置
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    
    # 全连接层配置
    fc_hidden_sizes: list = None
    dropout_rate: float = 0.5
    
    # 输出配置
    num_classes: int = 5 
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3]
        if self.fc_hidden_sizes is None:
            self.fc_hidden_sizes = [128, 64]


class DepAudioNet(nn.Module):
    
    def __init__(self, config: DepAudioNetConfig):
        super(DepAudioNet, self).__init__()
        self.config = config
        
        # 1D CNN层
        self.cnn_layers = self._build_cnn_layers()
        
        # 计算CNN输出维度
        self.cnn_output_size = self._calculate_cnn_output_size()
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 全连接层
        self.fc_layers = self._build_fc_layers()
        
        # 回归头 (用于总分预测)
        self.regression_head = nn.Linear(config.fc_hidden_sizes[-1], 1)
        
        # 分类头 (用于等级分类)
        self.classification_head = nn.Linear(config.fc_hidden_sizes[-1], config.num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _build_cnn_layers(self):
        layers = nn.ModuleList()
        
        in_channels = 1
        current_feature_size = self.config.input_features
        
        for i, out_channels in enumerate(self.config.cnn_channels):
            if current_feature_size < 4:
                break
                
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=min(self.config.cnn_kernel_sizes[i], current_feature_size // 2),
                stride=self.config.cnn_stride,
                padding=self.config.cnn_padding
            )
            
            # 批归一化
            bn = nn.BatchNorm1d(out_channels)
            
            # 计算卷积后的特征大小
            conv_output_size = current_feature_size + 2 * self.config.cnn_padding - (self.config.cnn_kernel_sizes[i] - 1)
            
            # 只有当特征足够大时才添加池化层
            if conv_output_size >= 2:
                pool = nn.MaxPool1d(kernel_size=min(2, conv_output_size), stride=min(2, conv_output_size))
                layers.append(nn.Sequential(conv, bn, nn.ReLU(), pool))
                current_feature_size = conv_output_size // 2
            else:
                layers.append(nn.Sequential(conv, bn, nn.ReLU()))
                current_feature_size = conv_output_size
            
            in_channels = out_channels
        
        return layers
    
    def _calculate_cnn_output_size(self):
        dummy_input = torch.randn(2, 1, self.config.input_features)
        
        with torch.no_grad():
            training_status = {}
            for name, module in self.cnn_layers.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    training_status[name] = module.training
                    module.eval()
            
            try:
                x = dummy_input
                for layer in self.cnn_layers:
                    x = layer(x)
                output_size = x.size(1)
            except RuntimeError as e:
                print(f"警告: CNN输出大小计算失败: {str(e)}")
                
                output_size = 32
            
            for name, module in self.cnn_layers.named_modules():
                if name in training_status:
                    if training_status[name]:
                        module.train()
        
        return output_size
    
    def _build_fc_layers(self):
        layers = []
        
        input_size = self.config.lstm_hidden_size
        
        for hidden_size in self.config.fc_hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate)
            ])
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x, return_features=False):
        batch_size, seq_len, input_dim = x.shape
        
        x = x.view(batch_size * seq_len, 1, input_dim)
        
        # 1D CNN特征提取
        try:
            for layer in self.cnn_layers:
                x = layer(x)
        except RuntimeError as e:
            print(f"警告: CNN前向传播失败: {str(e)}")
            x = torch.zeros(batch_size * seq_len, self.cnn_output_size, 1).to(x.device)
        
        if x.shape[2] == 0: 
            x = torch.zeros(batch_size * seq_len, x.shape[1], 1).to(x.device)
        
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        final_hidden = lstm_out[:, -1, :] 
        
        features = self.fc_layers(final_hidden)
        
        regression_output = self.regression_head(features)
        
        classification_output = self.classification_head(features)
        
        outputs = {
            'regression_output': regression_output,
            'classification_output': classification_output
        }
        
        if return_features:
            outputs['features'] = features
        
        return outputs
    
    def get_attention_weights(self, x):
        
        with torch.no_grad():
            batch_size, seq_len, input_dim = x.shape
            
            # CNN特征提取
            x_cnn = x.view(batch_size * seq_len, 1, input_dim)
            for layer in self.cnn_layers:
                x_cnn = layer(x_cnn)
            
            cnn_output_size = x_cnn.size(1)
            x_lstm = x_cnn.view(batch_size, seq_len, cnn_output_size)
            
            # LSTM输出
            lstm_out, _ = self.lstm(x_lstm)
            
            # 计算注意力权重 (使用能量函数)
            attention_weights = torch.softmax(
                torch.sum(lstm_out ** 2, dim=2), dim=1
            )
            
            return attention_weights


class DepAudioNetLoss(nn.Module):
    
    def __init__(self, regression_weight: float = 1.0, classification_weight: float = 1.0):
        super(DepAudioNetLoss, self).__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:

        regression_loss = self.mse_loss(
            outputs['regression_output'].squeeze(),
            targets['regression_target'].float()
        )
        
        classification_loss = self.ce_loss(
            outputs['classification_output'],
            targets['classification_target'].long()
        )
        
        total_loss = (
            self.regression_weight * regression_loss +
            self.classification_weight * classification_loss
        )
        
        return total_loss, {
            'regression_loss': regression_loss.item(),
            'classification_loss': classification_loss.item(),
            'total_loss': total_loss.item()
        }


def create_depaudionet_model(config: DepAudioNetConfig) -> DepAudioNet:

    input_size = config.input_features
    
    if input_size < 10: 
        config.cnn_channels = [16, 32]
        config.cnn_kernel_sizes = [1, 1]
        config.cnn_padding = 0
    elif input_size < 30: 
        config.cnn_channels = [32, 64]
        config.cnn_kernel_sizes = [2, 2]
        config.cnn_padding = 1
    
    print(f"模型配置: 输入特征={config.input_features}, CNN通道={config.cnn_channels}, 卷积核大小={config.cnn_kernel_sizes}")
    
    model = DepAudioNet(config)
    return model


if __name__ == "__main__":
    config = DepAudioNetConfig(
        input_features=74,
        sequence_length=100
    )
    
    model = create_depaudionet_model(config)
    
    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.sequence_length, config.input_features)
    
    outputs = model(dummy_input)
    
    print("模型架构:")
    print(model)
    
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"回归输出形状: {outputs['regression_output'].shape}")
    print(f"分类输出形状: {outputs['classification_output'].shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}") 