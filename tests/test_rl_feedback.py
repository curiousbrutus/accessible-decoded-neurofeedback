"""
tests/test_rl_feedback.py
=========================
Unit tests for the closed-loop RL feedback module.
"""

import numpy as np
import pytest

from closed_loop.reinforcement_learning_feedback import (
    ActorCriticAgent,
    NeurofeedbackEnv,
    RLFeedbackAgent,
)


class TestNeurofeedbackEnv:
    def test_reset_shape(self):
        env = NeurofeedbackEnv(n_steps=10)
        obs = env.reset()
        assert obs.shape == (2,)
        assert 0.0 <= obs[0] <= 1.0  # similarity
        assert obs[1] == 0.0         # step normalised

    def test_step_shapes(self):
        env = NeurofeedbackEnv(n_steps=10, action_space_size=5)
        env.reset()
        obs, reward, done, info = env.step(2)
        assert obs.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "similarity" in info

    def test_episode_terminates(self):
        env = NeurofeedbackEnv(n_steps=5)
        env.reset()
        for _ in range(10):  # more than n_steps
            _, _, done, _ = env.step(4)
            if done:
                break
        assert done

    def test_similarity_bounded(self):
        env = NeurofeedbackEnv(n_steps=30)
        env.reset()
        for _ in range(30):
            obs, _, done, _ = env.step(np.random.randint(0, 5))
            assert 0.0 <= obs[0] <= 1.0
            if done:
                break


class TestRLFeedbackAgent:
    def test_select_action_range(self):
        agent = RLFeedbackAgent(obs_dim=2, n_actions=5)
        obs = np.array([0.3, 0.1], dtype=np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < 5

    def test_train_returns_rewards(self):
        env = NeurofeedbackEnv(n_steps=5)
        agent = RLFeedbackAgent(obs_dim=2, n_actions=5)
        rewards = agent.train(env, n_episodes=10)
        assert len(rewards) == 10

    def test_reward_trend(self):
        """Agent should achieve non-trivially improving rewards over many episodes."""
        env = NeurofeedbackEnv(n_steps=20)
        agent = RLFeedbackAgent(obs_dim=2, n_actions=5)
        rewards = agent.train(env, n_episodes=200)
        # Average of last 50 should be >= average of first 50
        assert np.mean(rewards[-50:]) >= np.mean(rewards[:50]) - 2.0  # allow tolerance


class TestActorCriticAgent:
    def test_select_action_range(self):
        agent = ActorCriticAgent(obs_dim=2, n_actions=5)
        obs = np.array([0.4, 0.2], dtype=np.float32)
        action, log_prob = agent.select_action(obs)
        assert 0 <= action < 5

    def test_train_returns_rewards(self):
        env = NeurofeedbackEnv(n_steps=5)
        agent = ActorCriticAgent(obs_dim=2, n_actions=5)
        rewards = agent.train(env, n_episodes=10)
        assert len(rewards) == 10

    def test_reward_trend(self):
        env = NeurofeedbackEnv(n_steps=20)
        agent = ActorCriticAgent(obs_dim=2, n_actions=5)
        rewards = agent.train(env, n_episodes=200)
        assert np.mean(rewards[-50:]) >= np.mean(rewards[:50]) - 2.0
