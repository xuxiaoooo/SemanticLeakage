# CueGate: 稀疏语义门控网络用于Cue检测

CueGate是一个轻量级的深度学习模型，专门用于检测语音中的抑郁相关词（cue）。

---

## 🚀 快速开始

### 1. 训练模型

```bash
cd /home/a001/xuxiao/SemanticLeakage
python CueGate/CueGate/train.py --epochs 100 --batch-size 32
```

训练完成后，会在 `CueGate/CueGate/checkpoints/` 目录下生成：
- `best_model.pt` - 验证集上F1最高的模型
- `final_model.pt` - 最后一个epoch的模型
- `test_results.txt` - 测试集结果摘要

**训练过程会输出：**
- 每个epoch的训练/验证损失
- 验证集的Precision/Recall/F1
- 训练结束后自动在测试集上评估

**训练参数：**
```bash
python CueGate/CueGate/train.py \
    --epochs 100 \            # 训练轮数
    --batch-size 32 \         # 批次大小
    --lr 1e-3 \               # 学习率
    --segment-length 3.0 \    # 音频片段长度（秒）
    --seed 42                 # 随机种子
```

---

### 2. 查看训练结果

训练完成后，查看 `CueGate/CueGate/checkpoints/test_results.txt`：

```
======================================================================
CueGate Model - Test Set Results
======================================================================

Best Epoch: 45
Best Val F1: 0.8234

Test Set Metrics:
  Loss:      0.1234
  Accuracy:  0.9567
  Precision: 0.8123
  Recall:    0.7845
  F1 Score:  0.7982

======================================================================
```

**指标说明：**
- **Precision（精确率）**: 检测出的cue中，真正是cue的比例
- **Recall（召回率）**: 所有真实cue中，被检测出来的比例
- **F1 Score**: Precision和Recall的调和平均

---

### 3. 评估模型（在完整数据集上）

```bash
# 基本评估
python CueGate/CueGate/evaluate.py --checkpoint checkpoints/best_model.pt

# 自动寻找最佳阈值
python CueGate/CueGate/evaluate.py --checkpoint checkpoints/best_model.pt --tune-threshold

# 指定输出目录
python CueGate/CueGate/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --threshold 0.5 \
    --output-dir evaluation_results
```

**评估结果输出：**

评估完成后，会在 `CueGate/CueGate/evaluation_results/` 目录生成：

1. **`evaluation_summary.txt`** - 总体评估摘要
   ```
   ======================================================================
   CueGate Model Evaluation Summary
   ======================================================================
   
   Overall Metrics:
     Total samples:        189
     Total ground truth:   1234
     Total predictions:    1189
     True Positives:       987
     False Positives:      202
     False Negatives:      247
   
     Precision:            0.8301
     Recall:               0.7998
     F1 Score:             0.8147
   
   Per-Sample Statistics:
     Avg Precision:        0.8234 ± 0.1245
     Avg Recall:           0.7956 ± 0.1398
     Avg F1:               0.8089 ± 0.1187
   ```

2. **`per_sample_results.csv`** - 每个样本的详细结果
   - 包含每个音频文件的TP/FP/FN/Precision/Recall/F1

3. **`evaluation_results.json`** - JSON格式的完整结果

---

### 4. 使用模型进行推理

#### Python API

```python
from CueGate.CueGate import CueDetector

# 加载模型
detector = CueDetector("CueGate/CueGate/checkpoints/best_model.pt")

# 检测单个音频文件
results = detector.detect("path/to/audio.wav")
# 结果: [{'start': 1.2, 'end': 1.8, 'score': 0.95}, ...]

# 或传入波形数组
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
results = detector.detect(audio, sample_rate=16000)

# 打印结果
for i, cue in enumerate(results, 1):
    print(f"Cue {i}: [{cue['start']:.2f}s - {cue['end']:.2f}s] (score: {cue['score']:.3f})")

# 批量检测
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
batch_results = detector.batch_detect(audio_files)

# 获取帧级概率
result_dict = detector.detect_with_probs("audio.wav")
# result_dict['cue_probs']: 每帧的cue概率
# result_dict['frame_times']: 每帧对应的时间
# result_dict['gate']: 稀疏门控值
```

#### 命令行

```bash
# 检测单个音频
python CueGate/CueGate/inference.py audio.wav \
    --checkpoint checkpoints/best_model.pt \
    --threshold 0.5 \
    --output results.json

# 输出示例
# Found 3 cue(s):
#   1. [1.200s - 1.800s] score=0.950
#   2. [5.400s - 6.100s] score=0.873
#   3. [8.900s - 9.500s] score=0.821
```

---

## 📊 结果解读

### 训练过程监控

训练时会实时输出：
```
Epoch   1 | Train Loss: 0.4523 | Val Loss: 0.3821 | Val F1: 0.6234 | ...
Epoch   2 | Train Loss: 0.3912 | Val Loss: 0.3456 | Val F1: 0.6789 | ...
...
  → Saved best model (F1: 0.8234)  # 出现这个说明找到了更好的模型
```

### 评估指标

**帧级评估（训练时）：**
- 将音频分成10ms的帧，每帧判断是否是cue
- 适合训练时快速评估

**片段级评估（evaluate.py）：**
- 检测完整的cue时间区间
- 使用IoU (Intersection over Union) 匹配预测和真实标注
- 更接近实际应用场景

### 常见问题诊断

**Q: F1很低怎么办？**
- 检查训练样本数量（至少需要数百个cue样本）
- 尝试调整阈值（使用 `--tune-threshold`）
- 增加训练轮数或调整学习率

**Q: Precision高但Recall低？**
- 模型过于保守，漏检较多
- 降低检测阈值（如从0.5降到0.4）

**Q: Recall高但Precision低？**
- 模型过于激进，误报较多
- 提高检测阈值（如从0.5升到0.6）

---

## 🏗️ 模型架构

```
Input Waveform
      ↓
[Acoustic Stream] ← SincConv + Temporal Convs (局部特征)
      ↓
[Semantic Stream] ← Multi-Scale Dilated Convs (上下文)
      ↓
[Sparse Gate] ← 稀疏门控，显式建模cue的稀疏性
      ↓
[Classifier] + [Contrastive Head]
      ↓
Frame-level Cue Probabilities
```

**特点：**
- 参数量：~400K（轻量）
- 输入：原始波形（无需手工特征）
- 输出：帧级cue概率 + 自动聚合为时间区间

---

## 📁 文件结构

```
CueGate/CueGate/
├── __init__.py           # 模块入口
├── model.py              # 模型架构定义
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── inference.py          # 推理接口
├── checkpoints/          # 保存的模型（训练后生成）
│   ├── best_model.pt
│   ├── final_model.pt
│   └── test_results.txt
└── evaluation_results/   # 评估结果（评估后生成）
    ├── evaluation_summary.txt
    ├── per_sample_results.csv
    └── evaluation_results.json
```

---

## 💡 使用建议

1. **训练数据准备**：确保 `agent/outputs/E-DAIC/*/cue_detection.json` 存在且有标注
2. **首次训练**：使用默认参数，观察效果
3. **阈值调优**：训练后使用 `--tune-threshold` 找最佳阈值
4. **应用到新数据**：直接加载权重，调用 `detector.detect()`

---

## 📞 技术支持

如有问题，查看：
- 训练日志：查看终端输出
- 测试结果：`checkpoints/test_results.txt`
- 评估详情：`evaluation_results/evaluation_summary.txt`

