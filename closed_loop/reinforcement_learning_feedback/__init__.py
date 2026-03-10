"""
reinforcement_learning_feedback
================================
RL-based adaptive neurofeedback agents.

Classes
-------
NeurofeedbackEnv  : Gymnasium-compatible environment simulating a closed-loop session
RLFeedbackAgent   : Policy-gradient (REINFORCE) agent for adaptive neurofeedback
ActorCriticAgent  : Actor-critic (A2C-style) agent with adaptive reward shaping
"""

from closed_loop.reinforcement_learning_feedback.rl_feedback import (
    NeurofeedbackEnv,
    RLFeedbackAgent,
    ActorCriticAgent,
)

__all__ = ["NeurofeedbackEnv", "RLFeedbackAgent", "ActorCriticAgent"]
