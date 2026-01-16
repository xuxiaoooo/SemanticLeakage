"""
Default settings for the agent transcript pipeline.
This is a slimmed-down version that keeps ASR runs and prompt limits only.
"""

from pathlib import Path
from typing import Optional

# 使用 SiliconFlow 两个模型互相验证
ASR_RUNS = [
    {"alias": "A", "provider": "siliconflow", "model": "FunAudioLLM/SenseVoiceSmall", "repeat": 1},
    {"alias": "B", "provider": "siliconflow", "model": "TeleAI/TeleSpeechASR", "repeat": 1},
]

CHUNK_SETTINGS = {
    "chunk_size": 200,
    "overlap": 30,
    "input_token_soft_limit": 500,
    "estimated_output_soft_limit": 4000,
}

PROMPT_LIMITS = {
    "max_asr_text_chars": 12000,
    "critic_evidence_sample": 30,
    "critic_qa_sample": 50,
}


def load_api_keys(path: Optional[Path] = None) -> dict:
    """Load API keys from a JSON file (fallback to environment variables)."""
    import json
    import os

    cfg_path = path or (Path(__file__).parent / "api_keys.json")
    keys: dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                keys.update(json.load(f))
        except Exception:
            pass

    # 智谱 ASR API key
    if "ZHIPU_API_KEY" not in keys:
        env_key = os.environ.get("ZHIPU_API_KEY")
        if env_key:
            keys["ZHIPU_API_KEY"] = env_key
    # DeepSeek LLM API key
    if "DEEPSEEK_API_KEY" not in keys:
        env_key = os.environ.get("DEEPSEEK_API_KEY")
        if env_key:
            keys["DEEPSEEK_API_KEY"] = env_key
    # 兼容旧配置：SILICONFLOW_API_KEY
    if "SILICONFLOW_API_KEY" not in keys:
        env_key = os.environ.get("SILICONFLOW_API_KEY")
        if env_key:
            keys["SILICONFLOW_API_KEY"] = env_key

    def normalize(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return []

    keys["ZHIPU_API_KEY"] = normalize(keys.get("ZHIPU_API_KEY"))
    keys["DEEPSEEK_API_KEY"] = normalize(keys.get("DEEPSEEK_API_KEY"))
    keys["SILICONFLOW_API_KEY"] = normalize(keys.get("SILICONFLOW_API_KEY"))
    return keys
