import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import DALFNet, DALFConfig
from .data_loader import DALFDataLoader


def train_one_epoch(model: nn.Module, loader, device: torch.device, criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    for x, y in loader:
        x = x.to(device)            # [B,1,T]
        y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += loss.item()
    return total_loss / len(loader), total_mse / len(loader)


def evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_t = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
            pred = model(x)
            loss = criterion(pred, y_t)
            total_loss += loss.item()
            preds.extend(pred.cpu().numpy().flatten())
            labels.extend(np.asarray(y, dtype=np.float32).flatten())
    return total_loss / len(loader), np.array(preds), np.array(labels)


def main():
    print("DALF 模型训练")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # 数据路径 - DAIC
    # 优先从环境变量读取，如果没有则使用默认值
    base_root = Path(__file__).resolve().parents[3]
    audio_dir = os.environ.get('AUDIO_DIR', str(base_root / "data_segment" / "E-DAIC"))
    questionnaire_path = os.environ.get('QUESTIONNAIRE_PATH', str(base_root / "data" / "E-DAIC" / "Detailed_PHQ8_Labels.csv"))

    # 随机种子
    random_seed = int(os.environ.get('RANDOM_SEED', 42))
    print(f"使用随机种子: {random_seed}")

    # 数据加载器
    segment_length = 6.0
    data_loader = DALFDataLoader(
        audio_dir=audio_dir,
        questionnaire_path=questionnaire_path,
        target_task='total_score',
        segment_level=os.environ.get("SEGMENT_LEVEL", "primary"),
        audio_variant=os.environ.get("AUDIO_VARIANT", "pre"),
        batch_size=32,
        num_workers=4,
        segment_length=segment_length,
        random_state=random_seed,
        val_size=0.2,
        test_size=0.1,
        sample_rate=16000
    )

    info = data_loader.get_data_info()
    print("数据集信息:")
    print(f"  总样本数: {info['total_size']}")
    print(f"  训练集: {info['train_size']}")
    print(f"  验证集: {info['val_size']}")
    print(f"  测试集: {info['test_size']}")
    print(f"  标签范围: [{info['label_min']:.1f}, {info['label_max']:.1f}]")

    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()

    # 模型
    model = DALFNet(DALFConfig(
        sample_rate=16000,
        num_filters=64,
        gabor_kernel=401,
        pool_kernel=401,
        pool_stride=160,
        meb_blocks=4,
        mssa_hidden=128,
        head_channels=128,
        dropout=0.2
    )).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")

    # 训练配置（与论文设置接近）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    max_epochs = 100
    best_val = float('inf')
    best_state = None

    start = time.time()
    for epoch in range(max_epochs):
        tr_loss, tr_mse = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start
            print(f"Epoch {epoch+1}/{max_epochs}:")
            print(f"  训练损失: {tr_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {lr:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"已加载最佳模型 (验证损失: {best_val:.4f})")

    # 测试集评估
    print("\n评估 total_score 任务...")
    test_loss, preds_norm, labels_norm = evaluate(model, test_loader, device, criterion)
    preds = data_loader.inverse_transform_labels(preds_norm)
    labels = data_loader.inverse_transform_labels(labels_norm)

    # 计算指标
    mae = mean_absolute_error(labels, preds)
    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    r2 = r2_score(labels, preds)

    print(f"\n{'='*80}")
    print("最终结果 (DALF)")
    print(f"{'='*80}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"{'='*80}")
    print("训练完成!")


if __name__ == "__main__":
    main()

