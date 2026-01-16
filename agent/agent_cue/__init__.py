"""
Agent Cue: Depression cue detection from transcripts.
Uses LLM to directly detect depression-related words.
"""

from agent.agent_cue.pipeline import CueDetectionPipeline

__all__ = ["CueDetectionPipeline"]
