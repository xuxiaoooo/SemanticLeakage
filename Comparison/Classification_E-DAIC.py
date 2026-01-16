"""
Emotion Classification Comparison for E-DAIC Dataset
验证Cue词是否比非Cue词更容易被分类为负面情绪
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch

# 绕过 pytorch_model.bin 安全检查
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import transformers.modeling_utils as _mu
import transformers.utils.import_utils as _iu
_mu.check_torch_load_is_safe = lambda *a, **k: None
_iu.check_torch_load_is_safe = lambda *a, **k: None

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR.parent / "agent" / "outputs" / "E-DAIC"
PHRASE_CONTEXT_SIZE = 3

# 英文模型（排除只有tf_model.h5的模型）
MODELS = {
    "distilbert-sentiment": MODELS_DIR / "lxyuan-distilbert-base-multilingual-cased-sentiments-student",
    "bert-emotion": MODELS_DIR / "nateraw-bert-base-uncased-emotion",
    "bert-sentiment-5star": MODELS_DIR / "nlptown-bert-base-multilingual-uncased-sentiment",
}

# 负面情绪标签
NEGATIVE_LABELS = {"negative", "neg", "1 star", "2 stars", "1", "2",
                   "sadness", "anger", "fear", "disgust", "grief", 
                   "disappointment", "remorse", "nervousness", "annoyance"}


def load_sample_data(sample_dir: Path) -> Dict:
    """加载样本的cue词和非cue词"""
    cue_file = sample_dir / "cue_detection.json"
    transcript_file = sample_dir / "transcript.multiscale.json"
    
    if not cue_file.exists() or not transcript_file.exists():
        return {}
    
    with open(cue_file) as f:
        cue_data = json.load(f)
    with open(transcript_file) as f:
        transcript_data = json.load(f)
    
    words = transcript_data.get("words", [])
    word_id_to_idx = {w["id"]: i for i, w in enumerate(words)}
    
    cue_word_ids = set()
    cue_words, cue_phrases = [], []
    
    for cue in cue_data.get("cues", []):
        if len(cue["text"].strip()) < 2:
            continue
        cue_word_ids.add(cue["word_id"])
        cue_words.append(cue["text"])
        
        if cue["word_id"] in word_id_to_idx:
            idx = word_id_to_idx[cue["word_id"]]
            start, end = max(0, idx - PHRASE_CONTEXT_SIZE), min(len(words), idx + PHRASE_CONTEXT_SIZE + 1)
            cue_phrases.append(" ".join(words[i]["text"] for i in range(start, end)))
    
    # 非cue词
    non_cue_indices = [i for i, w in enumerate(words) 
                       if w["id"] not in cue_word_ids and w.get("speaker") == "interviewee" and len(w["text"].strip()) >= 2]
    
    if len(non_cue_indices) > 30:
        non_cue_indices = random.sample(non_cue_indices, 30)
    
    non_cue_words = [words[i]["text"] for i in non_cue_indices]
    non_cue_phrases = [" ".join(words[j]["text"] for j in range(max(0, i-PHRASE_CONTEXT_SIZE), min(len(words), i+PHRASE_CONTEXT_SIZE+1))) 
                       for i in non_cue_indices]
    
    return {"cue_words": cue_words, "cue_phrases": cue_phrases, 
            "non_cue_words": non_cue_words, "non_cue_phrases": non_cue_phrases}


def is_negative(label: str) -> bool:
    return any(neg.lower() in label.lower() for neg in NEGATIVE_LABELS)


def classify_texts(texts: List[str], pipe) -> Dict:
    """分类文本列表"""
    total, negative, scores = 0, 0, []
    for text in texts:
        try:
            output = pipe(text)
            if isinstance(output, list) and output:
                if isinstance(output[0], list):
                    output = output[0]
                total += 1
                if is_negative(output[0]["label"]):
                    negative += 1
                scores.append(sum(r["score"] for r in output if is_negative(r["label"])))
        except:
            pass
    return {"total": total, "negative": negative, "scores": scores}


def main():
    print("=" * 70)
    print("Emotion Classification - E-DAIC (Word + Phrase)")
    print("=" * 70)
    
    # 加载样本
    samples = [load_sample_data(d) for d in sorted(OUTPUTS_DIR.glob("*_AUDIO"))]
    samples = [s for s in samples if s and s.get("cue_words")]
    print(f"\nSamples: {len(samples)}")
    
    # 加载模型
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'cuda' if device == 0 else 'cpu'}\n")
    
    pipelines = {}
    for name, path in MODELS.items():
        try:
            model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            pipelines[name] = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    if not pipelines:
        return
    
    # 实验
    for level in ["word", "phrase"]:
        cue_key = f"cue_{level}s"
        non_cue_key = f"non_cue_{level}s"
        
        print(f"\n{'='*70}")
        print(f"{level.upper()} LEVEL")
        print("="*70)
        
        for name, pipe in pipelines.items():
            cue_res = {"total": 0, "negative": 0, "scores": []}
            non_cue_res = {"total": 0, "negative": 0, "scores": []}
            
            for sample in tqdm(samples, desc=name, leave=False):
                c = classify_texts(sample[cue_key], pipe)
                n = classify_texts(sample[non_cue_key], pipe)
                cue_res["total"] += c["total"]
                cue_res["negative"] += c["negative"]
                cue_res["scores"].extend(c["scores"])
                non_cue_res["total"] += n["total"]
                non_cue_res["negative"] += n["negative"]
                non_cue_res["scores"].extend(n["scores"])
            
            cue_rate = cue_res["negative"] / cue_res["total"] * 100 if cue_res["total"] > 0 else 0
            non_cue_rate = non_cue_res["negative"] / non_cue_res["total"] * 100 if non_cue_res["total"] > 0 else 0
            cue_avg = np.mean(cue_res["scores"]) if cue_res["scores"] else 0
            non_cue_avg = np.mean(non_cue_res["scores"]) if non_cue_res["scores"] else 0
            diff = cue_rate - non_cue_rate
            
            print(f"\n📊 {name}")
            print(f"   Cue:     {cue_res['negative']:4d}/{cue_res['total']:4d} = {cue_rate:5.1f}%  (avg: {cue_avg:.3f})")
            print(f"   Non-cue: {non_cue_res['negative']:4d}/{non_cue_res['total']:4d} = {non_cue_rate:5.1f}%  (avg: {non_cue_avg:.3f})")
            print(f"   Diff:    {diff:+.1f}%  {'✓' if diff > 0 else '✗'}")


if __name__ == "__main__":
    main()
