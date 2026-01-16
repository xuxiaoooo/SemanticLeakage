# AVEC2014 Cue Detection Analysis

## 概述
对 AVEC2014 数据集的抑郁相关词（Cue）检测结果进行统计分析。

## 使用方法

### 运行分析
```bash
cd StatisticalAnalysis/AVEC2014
python cue_analysis.py
```

### 输出文件
分析结果保存在 `outputs/` 目录下：

1. **cue_analysis_report.txt** - 完整的统计分析报告
   - 样本概览（总数、检测率）
   - Cue统计（总数、唯一词数、平均值）
   - 时长覆盖率统计
   - Top 20 高频词
   - 所有检测到的词列表

2. **cue_wordcloud.png** - 词云图
   - 可视化展示检测到的抑郁相关词
   - 词的大小表示出现频率

3. **word_frequencies.json** - 词频统计
   - JSON格式的完整词频数据
   - 按频率降序排列

4. **sample_statistics.json** - 每个样本的详细统计
   - 每个样本的检测数量
   - Cue时长和占比
   - 总词数

## 分析内容

### 1. 检测数量分布
统计每个样本检测到的Cue数量分布，包括：
- 检测数量范围
- 平均值和中位数
- 有/无检测的样本比例

### 2. Cue时长占比分布
分析Cue在音频中的时长占比，包括：
- 占比范围
- 平均值和中位数
- 分布直方图数据

## 数据来源
分析脚本从 `agent/outputs/AVEC2014/` 目录读取所有 `*_Freeform_video/cue_detection.json` 文件。

## 依赖
- Python 3.7+
- matplotlib
- wordcloud
- numpy

