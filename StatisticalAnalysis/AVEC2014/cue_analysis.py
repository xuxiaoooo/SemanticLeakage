"""
Cue Detection Analysis for AVEC2014 dataset.
分析检测出的抑郁相关词：词频统计、词云、时长占比、检测分布等。
"""

import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# 路径配置
OUTPUTS_DIR = Path(__file__).parent.parent.parent / "agent" / "outputs" / "AVEC2014"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_cue_detections():
    """加载所有cue_detection.json文件"""
    results = []
    for sample_dir in sorted(OUTPUTS_DIR.glob("*_Freeform_video")):
        cue_file = sample_dir / "cue_detection.json"
        if cue_file.exists():
            with open(cue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["sample_id"] = sample_dir.name.replace("_Freeform_video", "")
                results.append(data)
    return results


def analyze_words(results):
    """分析检测出的词"""
    all_words = []
    for r in results:
        for cue in r.get("cues", []):
            all_words.append(cue["text"].lower())
    
    word_counts = Counter(all_words)
    return word_counts, all_words


def generate_wordcloud(word_counts, output_path):
    """生成词云图 - 白色背景，紧凑，无标题"""
    if not word_counts:
        print("No words to generate wordcloud")
        return
    
    wc = WordCloud(
        width=800,
        height=300,
        background_color="white",
        max_words=200,
        min_font_size=10,
        max_font_size=100,
        relative_scaling=0.5,
        colormap="viridis",
        margin=5,
    )
    wc.generate_from_frequencies(word_counts)
    
    # 保存 - 紧凑无边距
    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Wordcloud saved to {output_path}")


def analyze_coverage(results):
    """分析cue时长占比"""
    coverage_data = []
    for r in results:
        stats = r.get("statistics", {})
        coverage_data.append({
            "sample_id": r["sample_id"],
            "cue_time_sec": stats.get("cue_time_coverage_sec", 0),
            "cue_ratio": stats.get("cue_time_coverage_ratio", 0),
            "cues_detected": stats.get("cues_detected", 0),
            "total_words": stats.get("total_words", 0),
        })
    return coverage_data


def plot_detection_distribution(coverage_data, output_path):
    """打印检测分布图的数值 - 每个样本检测出多少cue"""
    cue_counts = [d["cues_detected"] for d in coverage_data]
    
    # 统计有/无检测的样本数
    with_cues = sum(1 for c in cue_counts if c > 0)
    without_cues = len(cue_counts) - with_cues
    
    # 计算直方图数据
    max_cues = max(cue_counts)
    bins = range(0, max_cues + 2)
    hist, bin_edges = np.histogram(cue_counts, bins=bins)
    
    # 统计信息
    mean_cues = np.mean(cue_counts)
    median_cues = np.median(cue_counts)
    
    print("\n" + "="*80)
    print("Distribution 1: 检测数量分布 (Number of Cues Detected per Sample)")
    print("="*80)
    print(f"\n总样本数: {len(cue_counts)}")
    print(f"检测数量范围: {min(cue_counts)} - {max_cues}")
    print(f"平均值: {mean_cues:.2f}")
    print(f"中位数: {median_cues:.1f}")
    print(f"\n横坐标 (检测到的Cue数量) | 纵坐标 (样本数量)")
    print("-" * 50)
    
    for i in range(len(hist)):
        cue_num = int(bin_edges[i])
        sample_count = int(hist[i])
        if sample_count > 0:  # 只显示有数据的柱子
            print(f"{cue_num:3d} cues detected          | {sample_count:4d} samples")
    
    print(f"\n饼图数据:")
    print(f"  有检测到Cue的样本: {with_cues} ({with_cues/len(cue_counts)*100:.1f}%)")
    print(f"  没有检测到Cue的样本: {without_cues} ({without_cues/len(cue_counts)*100:.1f}%)")
    print("="*80)


def plot_coverage_ratio(coverage_data, output_path):
    """打印cue时长占比分布的数值"""
    ratios = [d["cue_ratio"] * 100 for d in coverage_data if d["cue_ratio"] > 0]
    
    if not ratios:
        print("No coverage data to plot")
        return
    
    # 计算直方图数据 (20 bins)
    hist, bin_edges = np.histogram(ratios, bins=20)
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    
    print("\n" + "="*80)
    print("Distribution 2: Cue时长占比分布 (Cue Time Coverage Ratio)")
    print("="*80)
    print(f"\n总样本数 (有Cue的): {len(ratios)}")
    print(f"占比范围: {min(ratios):.3f}% - {max(ratios):.3f}%")
    print(f"平均值: {mean_ratio:.3f}%")
    print(f"中位数: {median_ratio:.3f}%")
    print(f"\n横坐标 (Cue时长占比 %) | 纵坐标 (样本数量)")
    print("-" * 60)

    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        sample_count = int(hist[i])
        if sample_count > 0:  # 只显示有数据的柱子
            print(f"{bin_start:6.3f}% - {bin_end:6.3f}%    | {sample_count:4d} samples")

    print("="*80)


def generate_summary_report(results, word_counts, coverage_data):
    """生成汇总报告"""
    total_samples = len(results)
    samples_with_cues = sum(1 for d in coverage_data if d["cues_detected"] > 0)
    total_cues = sum(d["cues_detected"] for d in coverage_data)
    unique_words = len(word_counts)

    # Top 20 词
    top_words = word_counts.most_common(20)

    # 时长统计
    cue_times = [d["cue_time_sec"] for d in coverage_data]
    cue_ratios = [d["cue_ratio"] * 100 for d in coverage_data]

    report = f"""
================================================================================
                        CUE DETECTION ANALYSIS REPORT
                             AVEC2014 Dataset
================================================================================

1. OVERVIEW
   - Total samples analyzed: {total_samples}
   - Samples with detected cues: {samples_with_cues} ({samples_with_cues/total_samples*100:.1f}%)
   - Samples without cues: {total_samples - samples_with_cues} ({(total_samples-samples_with_cues)/total_samples*100:.1f}%)

2. CUE STATISTICS
   - Total cue instances detected: {total_cues}
   - Unique depression-related words: {unique_words}
   - Average cues per sample: {total_cues/total_samples:.2f}
   - Average cues per sample (with cues only): {(total_cues/samples_with_cues if samples_with_cues > 0 else 0):.2f}

3. TIME COVERAGE
   - Total cue time: {sum(cue_times):.2f} seconds
   - Average cue time per sample: {np.mean(cue_times):.2f} seconds
   - Average coverage ratio: {np.mean(cue_ratios):.3f}%
   - Max coverage ratio: {max(cue_ratios):.3f}%

4. TOP 20 DETECTED WORDS
"""
    for i, (word, count) in enumerate(top_words, 1):
        report += f"   {i:2d}. {word:15s} - {count:4d} occurrences\n"

    report += f"""
5. ALL DETECTED WORDS ({unique_words} unique)
   {', '.join(sorted(word_counts.keys()))}

================================================================================
"""
    return report


def main():
    print("Loading cue detection results...")
    results = load_all_cue_detections()
    print(f"Loaded {len(results)} samples")

    if not results:
        print("No cue detection results found!")
        return

    # 分析词频
    print("\nAnalyzing detected words...")
    word_counts, all_words = analyze_words(results)
    print(f"Total cue instances: {len(all_words)}")
    print(f"Unique words: {len(word_counts)}")

    # 生成词云
    print("\nGenerating wordcloud...")
    generate_wordcloud(word_counts, OUTPUT_DIR / "cue_wordcloud.png")

    # 分析覆盖率
    print("\nAnalyzing coverage...")
    coverage_data = analyze_coverage(results)

    # 绘制检测分布图
    print("\nPlotting detection distribution...")
    plot_detection_distribution(coverage_data, OUTPUT_DIR / "cue_detection_distribution.jpg")

    # 绘制时长占比分布
    print("\nPlotting coverage ratio distribution...")
    plot_coverage_ratio(coverage_data, OUTPUT_DIR / "cue_coverage_ratio.jpg")

    # 生成报告
    print("\nGenerating summary report...")
    report = generate_summary_report(results, word_counts, coverage_data)
    print(report)

    # 保存报告
    report_path = OUTPUT_DIR / "cue_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # 保存词频统计为JSON
    word_freq_path = OUTPUT_DIR / "word_frequencies.json"
    with open(word_freq_path, "w", encoding="utf-8") as f:
        json.dump(dict(word_counts.most_common()), f, indent=2, ensure_ascii=False)
    print(f"Word frequencies saved to {word_freq_path}")

    # 保存每个样本的统计为JSON
    sample_stats_path = OUTPUT_DIR / "sample_statistics.json"
    with open(sample_stats_path, "w", encoding="utf-8") as f:
        json.dump(coverage_data, f, indent=2, ensure_ascii=False)
    print(f"Sample statistics saved to {sample_stats_path}")


if __name__ == "__main__":
    main()


