"""
CueGate Evaluation Script

评估已训练的CueGate模型在整个数据集上的性能

功能：
1. 加载训练好的模型
2. 在每个音频文件上进行cue检测
3. 与ground truth标注对比，计算指标
4. 输出详细的评估报告

使用方式：
    python CueGate/CueGate/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from inference import CueDetector

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
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "E-DAIC"
AGENT_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "outputs" / "E-DAIC"


# ============================================================================
# Evaluation Metrics
# ============================================================================
def calculate_iou(pred_start: float, pred_end: float, 
                  gt_start: float, gt_end: float) -> float:
    """计算两个时间区间的IoU"""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0.0


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.3,
) -> Tuple[int, int, int]:
    """
    匹配预测和ground truth
    
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    matched_gt = set()
    tp = 0
    
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(
                pred['start'], pred['end'],
                gt['start'], gt['end']
            )
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    
    return tp, fp, fn


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """计算precision, recall, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ============================================================================
# Evaluation Functions
# ============================================================================
def load_ground_truth(cue_detection_path: Path) -> List[Dict]:
    """加载ground truth cue标注"""
    if not cue_detection_path.exists():
        return []
    
    with open(cue_detection_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cues = data.get('cues', [])
    return [{'start': c['start'], 'end': c['end'], 'text': c.get('text', '')} 
            for c in cues]


def evaluate_single_sample(
    detector: CueDetector,
    audio_path: Path,
    ground_truth: List[Dict],
    threshold: float = 0.5,
) -> Dict:
    """评估单个样本"""
    # 检测
    try:
        predictions = detector.detect(str(audio_path), threshold=threshold)
    except Exception as e:
        logger.warning(f"Detection failed for {audio_path}: {e}")
        predictions = []
    
    # 匹配和计算指标
    tp, fp, fn = match_predictions_to_ground_truth(predictions, ground_truth)
    metrics = calculate_metrics(tp, fp, fn)
    
    return {
        'num_predictions': len(predictions),
        'num_ground_truth': len(ground_truth),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        **metrics,
    }


def evaluate_dataset(
    checkpoint_path: str,
    threshold: float = 0.15,  # 降低默认阈值
    iou_threshold: float = 0.3,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    在整个数据集上评估模型
    
    Args:
        checkpoint_path: 模型权重路径
        threshold: 检测阈值
        iou_threshold: IoU阈值用于匹配
        output_dir: 输出目录
    
    Returns:
        评估结果字典
    """
    logger.info("=" * 70)
    logger.info("CueGate Model Evaluation")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Detection threshold: {threshold}")
    logger.info(f"IoU threshold: {iou_threshold}")
    
    # 加载检测器
    logger.info("\nLoading model...")
    detector = CueDetector(checkpoint_path, threshold=threshold)
    
    # 遍历所有样本
    sample_dirs = sorted(AGENT_OUTPUTS_DIR.glob("*_AUDIO"))
    logger.info(f"\nFound {len(sample_dirs)} samples\n")
    
    all_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for sample_dir in tqdm(sample_dirs, desc="Evaluating"):
        sample_id = sample_dir.name.replace("_AUDIO", "")
        
        # 音频路径
        audio_path = DATA_DIR / f"{sample_id}_P" / f"{sample_id}_AUDIO.wav"
        if not audio_path.exists():
            continue
        
        # 加载ground truth
        cue_path = sample_dir / "cue_detection.json"
        ground_truth = load_ground_truth(cue_path)
        
        # 评估
        result = evaluate_single_sample(detector, audio_path, ground_truth, threshold)
        result['sample_id'] = sample_id
        all_results.append(result)
        
        total_tp += result['tp']
        total_fp += result['fp']
        total_fn += result['fn']
    
    # 计算总体指标
    overall_metrics = calculate_metrics(total_tp, total_fp, total_fn)
    
    # 打印结果
    logger.info("\n" + "=" * 70)
    logger.info("Overall Results")
    logger.info("=" * 70)
    logger.info(f"Total samples:        {len(all_results)}")
    logger.info(f"Total ground truth:   {sum(r['num_ground_truth'] for r in all_results)}")
    logger.info(f"Total predictions:    {sum(r['num_predictions'] for r in all_results)}")
    logger.info(f"True Positives:       {total_tp}")
    logger.info(f"False Positives:      {total_fp}")
    logger.info(f"False Negatives:      {total_fn}")
    logger.info(f"\nPrecision:            {overall_metrics['precision']:.4f}")
    logger.info(f"Recall:               {overall_metrics['recall']:.4f}")
    logger.info(f"F1 Score:             {overall_metrics['f1']:.4f}")
    
    # 保存详细结果
    if output_dir is None:
        output_dir = SCRIPT_DIR / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存per-sample结果
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "per_sample_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nPer-sample results saved to {csv_path}")
    
    # 保存总结
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CueGate Model Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Detection threshold: {threshold}\n")
        f.write(f"IoU threshold: {iou_threshold}\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"  Total samples:        {len(all_results)}\n")
        f.write(f"  Total ground truth:   {sum(r['num_ground_truth'] for r in all_results)}\n")
        f.write(f"  Total predictions:    {sum(r['num_predictions'] for r in all_results)}\n")
        f.write(f"  True Positives:       {total_tp}\n")
        f.write(f"  False Positives:      {total_fp}\n")
        f.write(f"  False Negatives:      {total_fn}\n\n")
        f.write(f"  Precision:            {overall_metrics['precision']:.4f}\n")
        f.write(f"  Recall:               {overall_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:             {overall_metrics['f1']:.4f}\n\n")
        
        # 添加per-sample统计
        f.write("\nPer-Sample Statistics:\n")
        f.write(f"  Avg Precision:        {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}\n")
        f.write(f"  Avg Recall:           {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}\n")
        f.write(f"  Avg F1:               {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    logger.info(f"Summary saved to {summary_path}")
    
    # 保存JSON格式的结果
    json_path = output_dir / "evaluation_results.json"
    results_json = {
        'checkpoint': checkpoint_path,
        'threshold': threshold,
        'iou_threshold': iou_threshold,
        'overall_metrics': {
            'total_samples': len(all_results),
            'total_ground_truth': sum(r['num_ground_truth'] for r in all_results),
            'total_predictions': sum(r['num_predictions'] for r in all_results),
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            **overall_metrics,
        },
        'per_sample_stats': {
            'avg_precision': float(results_df['precision'].mean()),
            'avg_recall': float(results_df['recall'].mean()),
            'avg_f1': float(results_df['f1'].mean()),
            'std_precision': float(results_df['precision'].std()),
            'std_recall': float(results_df['recall'].std()),
            'std_f1': float(results_df['f1'].std()),
        },
        'per_sample_results': all_results,
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON results saved to {json_path}")
    
    return results_json


# ============================================================================
# Threshold Tuning
# ============================================================================
def find_best_threshold(
    checkpoint_path: str,
    thresholds: List[float] = None,
) -> Tuple[float, Dict]:
    """
    寻找最佳检测阈值
    
    Args:
        checkpoint_path: 模型权重路径
        thresholds: 要测试的阈值列表
    
    Returns:
        (best_threshold, best_results)
    """
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 降低阈值范围
    
    logger.info("\n" + "=" * 70)
    logger.info("Threshold Tuning")
    logger.info("=" * 70)
    
    best_threshold = None
    best_f1 = 0
    best_results = None
    
    for threshold in thresholds:
        logger.info(f"\nTesting threshold = {threshold}...")
        
        # 临时关闭详细日志
        logging.getLogger().setLevel(logging.WARNING)
        results = evaluate_dataset(
            checkpoint_path, 
            threshold=threshold,
            output_dir=None,
        )
        logging.getLogger().setLevel(logging.INFO)
        
        f1 = results['overall_metrics']['f1']
        logger.info(f"  F1 = {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_results = results
    
    logger.info(f"\nBest threshold: {best_threshold} (F1 = {best_f1:.4f})")
    
    return best_threshold, best_results


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CueGate model")
    parser.add_argument(
        "--checkpoint", type=str, 
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Detection threshold"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3,
        help="IoU threshold for matching"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--tune-threshold", action="store_true",
        help="Find best detection threshold"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = SCRIPT_DIR / checkpoint_path
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.tune_threshold:
        # 寻找最佳阈值
        best_threshold, best_results = find_best_threshold(str(checkpoint_path))
        
        # 用最佳阈值评估并保存
        logger.info(f"\nEvaluating with best threshold ({best_threshold})...")
        evaluate_dataset(
            str(checkpoint_path),
            threshold=best_threshold,
            iou_threshold=args.iou_threshold,
            output_dir=output_dir,
        )
    else:
        # 直接评估
        evaluate_dataset(
            str(checkpoint_path),
            threshold=args.threshold,
            iou_threshold=args.iou_threshold,
            output_dir=output_dir,
        )

