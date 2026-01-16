"""
Agent ASR: Speech-to-text transcription pipeline.
Handles: Preprocess -> ASR Ensemble -> Reconcile -> Role Attribution -> MultiScale Transcript
"""

from agent.agent_asr.pipeline import TranscriptPipeline

__all__ = ["TranscriptPipeline"]
