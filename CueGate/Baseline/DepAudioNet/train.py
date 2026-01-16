
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple, Optional

from .depaudionet_model import DepAudioNet, DepAudioNetConfig, DepAudioNetLoss, create_depaudionet_model
from .data_loader import DepAudioNetDataLoader

def train_depaudionet(
    audio_dir: str,
    questionnaire_path: str,
    target_task: str = 'total_score',
    segment_level: str = "primary",
    audio_variant: str = "pre",
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[DepAudioNet, Dict[str, Any]]:
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"使用设备: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 加载数据
    # 从环境变量获取seed，如果没有则使用默认值
    random_seed = int(os.environ.get('RANDOM_SEED', 42))
    
    data_loader = DepAudioNetDataLoader(
        audio_dir=audio_dir,
        questionnaire_path=questionnaire_path,
        target_task=target_task,
        segment_level=segment_level,
        audio_variant=audio_variant,
        batch_size=batch_size,
        num_workers=num_workers,
        segment_length=3.0,
        random_state=random_seed
    )
    
    if verbose:
        data_info = data_loader.get_data_info()
        print(f"数据集信息:")
        print(f"  总样本数: {data_info['total_size']}")
        print(f"  训练集: {data_info['train_size']}")
        print(f"  验证集: {data_info['val_size']}")
        print(f"  测试集: {data_info['test_size']}")
        print(f"  标签范围: [{data_info['label_min']:.1f}, {data_info['label_max']:.1f}]")
    
    # 创建模型
    config = DepAudioNetConfig(
        input_features=1,
        sequence_length=100,
        cnn_channels=[32, 64, 128],
        cnn_kernel_sizes=[3, 3, 3],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        lstm_dropout=0.3,
        fc_hidden_sizes=[128, 64],
        dropout_rate=0.5,
        num_classes=5
    )
    
    model = create_depaudionet_model(config)
    model = model.to(device)
    
    if verbose:
        print(f"模型架构:")
        print(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建损失函数
    criterion = DepAudioNetLoss(regression_weight=1.0, classification_weight=0.2)
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 获取数据加载器
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    # 最佳模型状态
    best_val_loss = float('inf')
    best_model_state = None
    
    # 开始训练
    if verbose:
        print(f"\n开始训练 {target_task} 任务...")
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_regression_loss = 0.0
        train_classification_loss = 0.0
        
        all_train_preds = []
        all_train_labels = []
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            # 准备目标
            targets = {
                'regression_target': labels,
                'classification_target': torch.zeros_like(labels, dtype=torch.long)  # 临时分类目标
            }
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(features)
            
            # 计算损失
            loss, loss_dict = criterion(outputs, targets)
            
            # 检查是否有NaN
            if torch.isnan(loss):
                print(f"警告: 训练损失为NaN在Epoch {epoch}, Batch {batch_idx}")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累积损失
            train_loss += loss.item()
            train_regression_loss += loss_dict['regression_loss']
            train_classification_loss += loss_dict['classification_loss']
            
            # 收集预测和标签
            all_train_preds.extend(outputs['regression_output'].detach().cpu().numpy().flatten())
            all_train_labels.extend(labels.cpu().numpy().flatten())
        
        # 计算训练集指标
        train_mae = mean_absolute_error(all_train_labels, all_train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_regression_loss = 0.0
        val_classification_loss = 0.0
        
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                # 准备目标
                targets = {
                    'regression_target': labels,
                    'classification_target': torch.zeros_like(labels, dtype=torch.long)  # 临时分类目标
                }
                
                # 前向传播
                outputs = model(features)
                
                # 计算损失
                loss, loss_dict = criterion(outputs, targets)
                
                # 累积损失
                val_loss += loss.item()
                val_regression_loss += loss_dict['regression_loss']
                val_classification_loss += loss_dict['classification_loss']
                
                # 收集预测和标签
                all_val_preds.extend(outputs['regression_output'].detach().cpu().numpy().flatten())
                all_val_labels.extend(labels.cpu().numpy().flatten())
        
        # 计算验证集指标
        val_mae = mean_absolute_error(all_val_labels, all_val_preds)
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        train_regression_loss /= len(train_loader)
        train_classification_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_regression_loss /= len(val_loader)
        val_classification_loss /= len(val_loader)
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # 打印进度
        if verbose and (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (epoch + 1) * (max_epochs - epoch - 1)
            
            print(f"Epoch {epoch+1}/{max_epochs}:")
            print(f"  训练损失: {train_loss:.4f} (回归: {train_regression_loss:.4f}, 分类: {train_classification_loss:.4f})")
            print(f"  验证损失: {val_loss:.4f} (回归: {val_regression_loss:.4f}, 分类: {val_classification_loss:.4f})")
            print(f"  训练MAE: {train_mae:.4f}, 验证MAE: {val_mae:.4f}")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  用时: {elapsed_time:.1f}s, 预计剩余: {eta:.1f}s")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"已加载最佳模型 (验证损失: {best_val_loss:.4f})")
    
    # 在测试集上评估模型
    if verbose:
        print(f"\n评估 {target_task} 任务...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            
            # 前向传播
            outputs = model(features)
            
            # 收集预测和标签
            all_preds.extend(outputs['regression_output'].cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    # 反标准化
    all_preds_orig = data_loader.inverse_transform_labels(np.array(all_preds))
    all_labels_orig = data_loader.inverse_transform_labels(np.array(all_labels))
    
    # 计算指标
    mae = mean_absolute_error(all_labels_orig, all_preds_orig)
    rmse = np.sqrt(mean_squared_error(all_labels_orig, all_preds_orig))
    r2 = r2_score(all_labels_orig, all_preds_orig)
    
    # 保存结果
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'history': history
    }
    
    if verbose:
        print(f"\n{target_task} 任务结果:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
    
    return model, results


def main():
    """主函数"""
    print("DepAudioNet模型训练")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 数据路径 - DAIC数据集
    # 优先从环境变量读取，如果没有则使用默认值
    base_root = Path(__file__).resolve().parents[3]
    audio_dir = os.environ.get('AUDIO_DIR', str(base_root / "data_segment" / "E-DAIC"))
    questionnaire_path = os.environ.get('QUESTIONNAIRE_PATH', str(base_root / "data" / "E-DAIC" / "Detailed_PHQ8_Labels.csv"))
    
    # 创建保存目录
    save_dir = os.environ.get('SAVE_DIR', str(base_root / "runs" / "DepAudioNet"))
    os.makedirs(save_dir, exist_ok=True)
    
    # 从环境变量获取seed
    random_seed = int(os.environ.get('RANDOM_SEED', 42))
    print(f"使用随机种子: {random_seed}")
    
    # 定义任务 - 只运行total_score任务
    task = 'total_score'
    
    print(f"\n{'='*60}")
    print(f"训练任务: {task}")
    print(f"{'='*60}")
    
    # 训练模型
    model, task_results = train_depaudionet(
        audio_dir=audio_dir,
        questionnaire_path=questionnaire_path,
        target_task=task,
        segment_level=os.environ.get("SEGMENT_LEVEL", "primary"),
        audio_variant=os.environ.get("AUDIO_VARIANT", "pre"),
        batch_size=32,
        num_workers=4,
        max_epochs=100,
        learning_rate=0.001,
        device=device,
        save_dir=save_dir,
        verbose=True
    )
    
    # 输出结果（用于解析）
    print(f"\n{'='*80}")
    print("最终结果 (DepAudioNet)")
    print(f"{'='*80}")
    print(f"MAE:  {task_results['MAE']:.4f}")
    print(f"RMSE: {task_results['RMSE']:.4f}")
    print(f"R2:   {task_results['R2']:.4f}")
    print(f"{'='*80}")
    print("训练完成!")


if __name__ == "__main__":
    main() 
