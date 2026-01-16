
import os
import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .stfn_model import STFNModel
from .data_loader import STFNDataLoader
import warnings
warnings.filterwarnings('ignore')


def train_model(model, train_loader, val_loader, device, config):

    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50, 
        T_mult=2,
        eta_min=1e-6 
    )
    
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"训练配置：")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  最大轮数: {config['max_epochs']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  HCPC权重: {config['hcpc_weight']}")
    print(f"  音频长度: {config['audio_length']}秒")
    
    for epoch in range(config['max_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_hcpc_loss = 0.0
        
        for batch_idx, (audios, labels) in enumerate(train_loader):
            audios = audios.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            predictions, hcpc_loss = model(audios)

            if torch.isnan(predictions).any():
                print(f"训练时发现NaN预测值在batch {batch_idx}, epoch {epoch}")
                print(f"NaN数量: {torch.isnan(predictions).sum().item()}")
                continue
            
            if torch.isnan(hcpc_loss):
                print(f"训练时发现NaN HCPC损失在batch {batch_idx}, epoch {epoch}")

                continue
            
            mse = mse_loss(predictions, labels)
            
            if torch.isnan(mse):
                print(f"训练时发现NaN MSE损失在batch {batch_idx}, epoch {epoch}")
                continue
                
            total_loss = mse + config['hcpc_weight'] * hcpc_loss
            
            if torch.isnan(total_loss):
                print(f"训练时发现NaN总损失在batch {batch_idx}, epoch {epoch}")
                continue

            total_loss.backward()
            
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"训练时发现NaN梯度在参数 {name}, batch {batch_idx}, epoch {epoch}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                optimizer.zero_grad()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            train_mse_loss += mse.item()
            train_hcpc_loss += hcpc_loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audios, labels in val_loader:
                audios = audios.to(device)
                labels = labels.to(device)
                
                predictions, hcpc_loss = model(audios)
                mse = mse_loss(predictions, labels)
                total_loss = mse + config['hcpc_weight'] * hcpc_loss
                val_loss += total_loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_hcpc_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}/{config["max_epochs"]}:')
            print(f'  训练损失: {train_loss:.4f} (MSE: {train_mse_loss:.4f}, HCPC: {train_hcpc_loss:.4f})')
            print(f'  验证损失: {val_loss:.4f}')
            print(f'  学习率: {current_lr:.2e}')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型 (验证损失: {best_val_loss:.4f})")
    
    return model


def evaluate_model(model, test_loader, data_loader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for audios, labels in test_loader:
            audios = audios.to(device)
            labels = labels.to(device)
            
            predictions, _ = model(audios)
            
            # 检查模型预测是否包含NaN
            if torch.isnan(predictions).any():
                print(f"警告: 模型预测包含NaN值!")
                print(f"NaN预测数量: {torch.isnan(predictions).sum().item()}")

                predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"标准化预测中的NaN数量: {np.isnan(all_predictions).sum()}")
    print(f"标准化标签中的NaN数量: {np.isnan(all_labels).sum()}")
    
    # 反标准化
    try:
        all_predictions_orig = data_loader.inverse_transform_labels(all_predictions)
        all_labels_orig = data_loader.inverse_transform_labels(all_labels)
    except Exception as e:
        print(f"反标准化过程中出错: {e}")
        raise e
    
    print(f"反标准化预测中的NaN数量: {np.isnan(all_predictions_orig).sum()}")
    print(f"反标准化标签中的NaN数量: {np.isnan(all_labels_orig).sum()}")

    if np.isnan(all_predictions_orig).any() or np.isnan(all_labels_orig).any():
        print(f"预测样本值: {all_predictions_orig[:10]}")
        print(f"标签样本值: {all_labels_orig[:10]}")
        
        valid_mask = ~(np.isnan(all_predictions_orig) | np.isnan(all_labels_orig))
        if valid_mask.sum() == 0:
            print("错误: 所有值都是NaN!")
            return float('nan'), float('nan'), float('nan')
        
        all_predictions_orig = all_predictions_orig[valid_mask]
        all_labels_orig = all_labels_orig[valid_mask]
        print(f"移除NaN后剩余有效样本数: {len(all_predictions_orig)}")
    
    mae = mean_absolute_error(all_labels_orig, all_predictions_orig)
    rmse = np.sqrt(mean_squared_error(all_labels_orig, all_predictions_orig))
    r2 = r2_score(all_labels_orig, all_predictions_orig)
    
    return mae, rmse, r2


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 数据路径 - DAIC数据集
    # 优先从环境变量读取，如果没有则使用默认值
    base_root = Path(__file__).resolve().parents[3]
    audio_dir = os.environ.get('AUDIO_DIR', str(base_root / "data_segment" / "E-DAIC"))
    questionnaire_path = os.environ.get('QUESTIONNAIRE_PATH', str(base_root / "data" / "E-DAIC" / "Detailed_PHQ8_Labels.csv"))
    
    # 从环境变量获取seed
    random_seed = int(os.environ.get('RANDOM_SEED', 42))
    print(f"使用随机种子: {random_seed}")
    
    # 定义任务 - 只运行total_score任务
    task = 'total_score'
    
    print(f"\n{'='*60}")
    print(f"训练任务: {task}")
    print(f"{'='*60}")
    
    data_loader = STFNDataLoader(
        audio_dir=audio_dir,
        questionnaire_path=questionnaire_path,
        target_task=task,
        segment_level=os.environ.get("SEGMENT_LEVEL", "primary"),
        audio_variant=os.environ.get("AUDIO_VARIANT", "pre"),
        batch_size=20, 
        num_workers=4,
        segment_length=3.0, 
        random_state=random_seed,
        test_size=0.1,
        val_size=0.2
    )
    
    # 数据集信息
    data_info = data_loader.get_data_info()
    print(f"数据集信息:")
    print(f"  总样本数: {data_info['total_size']}")
    print(f"  训练集: {data_info['train_size']}")
    print(f"  验证集: {data_info['val_size']}")
    print(f"  测试集: {data_info['test_size']}")
    print(f"  标签范围: [{data_info['label_min']:.1f}, {data_info['label_max']:.1f}]")
    
    model = STFNModel(
        input_dim=1,
        dropout=0.5, 
        prediction_steps=1 
    )
    
    model_info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  总参数数: {model_info['total_parameters']:,}")
    print(f"  可训练参数: {model_info['trainable_parameters']:,}")
    print(f"  各组件参数:")
    for component, params in model_info['model_components'].items():
        print(f"    {component}: {params:,}")
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    training_config = model.training_config
    
    print(f"\n开始训练 {task} 任务...")
    model = train_model(model, train_loader, val_loader, device, training_config)
    
    # 评估模型
    print(f"\n评估 {task} 任务...")
    mae, rmse, r2 = evaluate_model(model, test_loader, data_loader, device)
    
    # 输出结果（用于解析）
    print(f"\n{'='*80}")
    print("最终结果 (STFN)")
    print(f"{'='*80}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"{'='*80}")
    print("训练完成!")


if __name__ == "__main__":
    main() 
