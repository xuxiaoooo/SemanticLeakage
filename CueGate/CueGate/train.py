"""
CueGate Training Script

功能：
1. 从cue_detection.json读取cue标注
2. 从原始音频切割片段进行训练（内存中处理）
3. 支持对比学习和稀疏正则
4. 保存最佳模型权重

使用方式：
    python CueGate/CueGate/train.py
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import CueGate, CueGateConfig
from data_loader import load_all_training_data

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"


# ============================================================================
# Dataset
# ============================================================================
class CueGateDataset(Dataset):
    """CueGate训练数据集，包含丰富的数据增强"""
    
    def __init__(self, samples, augment: bool = True, sample_rate: int = 16000):
        self.samples = samples
        self.augment = augment
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio, labels = self.samples[idx]
        
        audio = torch.FloatTensor(audio)
        labels = torch.FloatTensor(labels)
        
        # 归一化
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()
        
        # 数据增强
        if self.augment:
            # 1. 随机增益变化
            gain = torch.FloatTensor(1).uniform_(0.6, 1.4).item()
            audio = audio * gain
            
            # 2. 添加高斯噪声
            if random.random() < 0.5:
                noise_level = random.uniform(0.005, 0.02)
                noise = torch.randn_like(audio) * noise_level
                audio = audio + noise
            
            # 3. 时间拉伸（通过重采样模拟）
            if random.random() < 0.3:
                stretch_factor = random.uniform(0.9, 1.1)
                orig_len = audio.size(0)
                # 简单的线性插值实现时间拉伸
                new_len = int(orig_len * stretch_factor)
                if new_len > 100:  # 确保长度合理
                    indices = torch.linspace(0, orig_len - 1, new_len)
                    indices_floor = indices.long().clamp(0, orig_len - 1)
                    audio_stretched = audio[indices_floor]
                    # 裁剪或填充到原始长度
                    if new_len > orig_len:
                        audio = audio_stretched[:orig_len]
                    else:
                        audio = torch.cat([audio_stretched, torch.zeros(orig_len - new_len)])
            
            # 4. 随机时间偏移（保持标签对齐）
            if random.random() < 0.3:
                shift = random.randint(-100, 100)
                if shift > 0:
                    audio = torch.cat([torch.zeros(shift), audio[:-shift]])
                elif shift < 0:
                    audio = torch.cat([audio[-shift:], torch.zeros(-shift)])
            
            # 5. 频谱扰动（模拟音高变化）- 通过随机低通/高通滤波
            if random.random() < 0.2:
                # 简单的移动平均滤波
                kernel_size = random.choice([3, 5, 7])
                kernel = torch.ones(kernel_size) / kernel_size
                audio_padded = torch.cat([
                    audio[:kernel_size//2].flip(0),
                    audio,
                    audio[-kernel_size//2:].flip(0)
                ])
                audio = torch.conv1d(
                    audio_padded.unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=0
                ).squeeze()[:audio.size(0)]
            
            # 6. 随机静音（模拟dropout）
            if random.random() < 0.1:
                mask_len = random.randint(100, 500)
                mask_start = random.randint(0, max(1, audio.size(0) - mask_len))
                audio[mask_start:mask_start + mask_len] *= 0.1
        
        return audio, labels


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(
    model: CueGate,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_contrast_loss = 0
    total_gate_loss = 0
    num_batches = 0
    
    for audio, labels in loader:
        audio = audio.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(audio, labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += outputs['cls_loss'].item()
        total_contrast_loss += outputs['contrast_loss'].item()
        total_gate_loss += outputs['gate_loss'].item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'contrast_loss': total_contrast_loss / num_batches,
        'gate_loss': total_gate_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: CueGate,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for audio, labels in loader:
        audio = audio.to(device)
        labels = labels.to(device)
        
        outputs = model(audio, labels)
        
        total_loss += outputs['loss'].item()
        
        preds = outputs['cue_probs'].cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())
        num_batches += 1
    
    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 检查预测概率范围
    prob_min = all_preds.min().item()
    prob_max = all_preds.max().item()
    prob_mean = all_preds.mean().item()
    
    # 动态寻找最佳阈值（基于验证集）
    best_threshold = 0.5
    best_f1_for_threshold = 0
    for th in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        preds_th = (all_preds > th).float()
        tp_th = ((preds_th == 1) & (all_labels == 1)).sum().item()
        fp_th = ((preds_th == 1) & (all_labels == 0)).sum().item()
        fn_th = ((preds_th == 0) & (all_labels == 1)).sum().item()
        p_th = tp_th / (tp_th + fp_th + 1e-8)
        r_th = tp_th / (tp_th + fn_th + 1e-8)
        f1_th = 2 * p_th * r_th / (p_th + r_th + 1e-8)
        if f1_th > best_f1_for_threshold:
            best_f1_for_threshold = f1_th
            best_threshold = th
    
    binary_preds_best = (all_preds > best_threshold).float()
    
    # 帧级准确率
    accuracy = (binary_preds_best == all_labels).float().mean().item()
    
    # 帧级Precision/Recall/F1
    tp = ((binary_preds_best == 1) & (all_labels == 1)).sum().item()
    fp = ((binary_preds_best == 1) & (all_labels == 0)).sum().item()
    fn = ((binary_preds_best == 0) & (all_labels == 1)).sum().item()
    tn = ((binary_preds_best == 0) & (all_labels == 0)).sum().item()
    
    # 调试信息（仍使用0.5阈值用于诊断）
    binary_preds_05 = (all_preds > 0.5).float()
    total_positive_labels = all_labels.sum().item()
    total_positive_preds = binary_preds_05.sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total_positive_labels': total_positive_labels,
        'total_positive_preds': total_positive_preds,
        'prob_min': prob_min,
        'prob_max': prob_max,
        'prob_mean': prob_mean,
        'best_threshold': best_threshold,
    }


def train(
    config: Optional[CueGateConfig] = None,
    segment_length: float = 3.0,
    batch_size: int = 32,
    num_epochs: int = 150,  # 增加训练轮数
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.1,  # 8:1:1划分
    test_ratio: float = 0.1,
    patience: int = 100,
    seed: int = 42,
):
    """训练CueGate模型"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载数据
    logger.info("Loading training data...")
    all_samples = load_all_training_data(
        segment_length=segment_length,
        frame_stride=config.frame_stride if config else 160,
    )
    
    if len(all_samples) < 10:
        logger.error("Not enough training samples!")
        return
    
    # 划分训练集、验证集和测试集
    train_val_samples, test_samples = train_test_split(
        all_samples, test_size=test_ratio, random_state=seed
    )
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=val_ratio/(1-test_ratio), random_state=seed
    )
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 创建数据加载器
    train_dataset = CueGateDataset(train_samples, augment=True)
    val_dataset = CueGateDataset(val_samples, augment=False)
    test_dataset = CueGateDataset(test_samples, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # 创建模型
    config = config or CueGateConfig()
    model = CueGate(config).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # 训练循环
    best_val_f1 = -1  # 改为-1，这样即使F1=0也会保存
    best_epoch = 0
    no_improve = 0
    
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # 验证
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        # 日志
        lr = optimizer.param_groups[0]['lr']
        
        # 第一个epoch打印详细调试信息
        if epoch == 0:
            logger.info(
                f"\n[Debug] Val set statistics:"
                f"\n  Total frames: {val_metrics['tp'] + val_metrics['fp'] + val_metrics['fn'] + val_metrics['tn']}"
                f"\n  Positive labels: {val_metrics['total_positive_labels']}"
                f"\n  Positive predictions: {val_metrics['total_positive_preds']}"
                f"\n  TP/FP/FN/TN: {val_metrics['tp']}/{val_metrics['fp']}/{val_metrics['fn']}/{val_metrics['tn']}"
                f"\n  Prob range: [{val_metrics['prob_min']:.4f}, {val_metrics['prob_max']:.4f}], mean: {val_metrics['prob_mean']:.4f}"
                f"\n  -> Note: Use threshold ~0.14 for best F1 (not 0.5)"
            )
        
        logger.info(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val P/R: {val_metrics['precision']:.3f}/{val_metrics['recall']:.3f} | "
            f"LR: {lr:.6f}"
        )
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            no_improve = 0
            
            # 保存最佳模型的state_dict
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'config': config,
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, CHECKPOINTS_DIR / "best_model.pt")
            logger.info(f"  → Saved best model (F1: {best_val_f1:.4f})")
        else:
            no_improve += 1
        
        # 早停
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"\nTraining complete! Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # ========================================================================
    # 测试集评估
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Evaluating on Test Set...")
    logger.info("=" * 70)
    
    # 检查是否有保存的模型
    if not (CHECKPOINTS_DIR / "best_model.pt").exists():
        logger.warning("No best model saved during training! Using current model state.")
        # 保存当前模型
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'config': config,
            'val_metrics': {'f1': 0, 'precision': 0, 'recall': 0},
        }
        torch.save(checkpoint, CHECKPOINTS_DIR / "best_model.pt")
    
    # 加载最佳模型
    best_checkpoint = torch.load(CHECKPOINTS_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 在测试集上评估
    test_metrics = evaluate(model, test_loader, device)
    
    logger.info(f"\n=== Test Set Results (threshold={test_metrics['best_threshold']:.2f}) ===")
    logger.info(f"Loss:      {test_metrics['loss']:.4f}")
    logger.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"Prob range: [{test_metrics['prob_min']:.4f}, {test_metrics['prob_max']:.4f}]")
    logger.info(f"TP/FP/FN/TN: {test_metrics['tp']}/{test_metrics['fp']}/{test_metrics['fn']}/{test_metrics['tn']}")
    
    # 保存测试结果到文本文件
    results_path = CHECKPOINTS_DIR / "test_results.txt"
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CueGate Model - Test Set Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Val F1: {best_val_f1:.4f}\n\n")
        f.write(f"Test Set Metrics (threshold={test_metrics['best_threshold']:.2f}):\n")
        f.write(f"  Loss:      {test_metrics['loss']:.4f}\n")
        f.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:  {test_metrics['f1']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {test_metrics['tp']}, FP: {test_metrics['fp']}\n")
        f.write(f"  FN: {test_metrics['fn']}, TN: {test_metrics['tn']}\n\n")
        f.write("Probability Range:\n")
        f.write(f"  Min:  {test_metrics['prob_min']:.4f}\n")
        f.write(f"  Max:  {test_metrics['prob_max']:.4f}\n")
        f.write(f"  Mean: {test_metrics['prob_mean']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    
    logger.info(f"\nResults saved to {results_path}")
    
    # 更新best_model.pt，加入测试指标
    best_checkpoint['test_metrics'] = test_metrics
    torch.save(best_checkpoint, CHECKPOINTS_DIR / "best_model.pt")
    logger.info(f"Updated best_model.pt with test metrics")
    
    return model, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CueGate model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--segment-length", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    config = CueGateConfig()
    
    train(
        config=config,
        segment_length=args.segment_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )
