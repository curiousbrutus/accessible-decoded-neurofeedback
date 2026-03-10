"""
closed_loop
===========
Closed-loop neurofeedback module with reinforcement learning algorithms.

Sub-packages
------------
reinforcement_learning_feedback : RL agents that adapt feedback signals
"""

from closed_loop.reinforcement_learning_feedback import RLFeedbackAgent

__all__ = ["RLFeedbackAgent"]
