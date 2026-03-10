"""
experiments
===========
Experimental task implementations for neurofeedback studies.

Sub-packages
------------
perception_task  : Visual discrimination / perception task
confidence_task  : Confidence judgement and metacognitive task
"""

from experiments.perception_task import PerceptionTask
from experiments.confidence_task import ConfidenceTask

__all__ = ["PerceptionTask", "ConfidenceTask"]
