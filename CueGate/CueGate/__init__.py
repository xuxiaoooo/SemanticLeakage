"""
CueGate: 稀疏语义门控网络用于Cue检测

使用方式：
    from CueGate.CueGate import CueDetector, CueGate, CueGateConfig
    
    # 推理
    detector = CueDetector("checkpoints/best_model.pt")
    cues = detector.detect("audio.wav")
    
    # 训练
    from CueGate.CueGate.train import train
    train(num_epochs=100)
"""

from .model import CueGate, CueGateConfig, create_model
from .inference import CueDetector, load_detector

__all__ = [
    "CueGate",
    "CueGateConfig", 
    "create_model",
    "CueDetector",
    "load_detector",
]

