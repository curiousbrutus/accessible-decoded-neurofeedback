"""
tests/test_pipeline.py
======================
Unit tests for the real-time neurofeedback pipeline.
"""

import numpy as np
import pytest

from closed_loop.reinforcement_learning_feedback import ActorCriticAgent, NeurofeedbackEnv
from models.fmri_decoder import FMRIDecoder
from pipelines import RealtimeNeurofeedbackPipeline


@pytest.fixture
def localiser_data():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 50))
    y = rng.integers(0, 2, size=100)
    X[y == 1, :5] += 1.5
    return X, y


class TestRealtimeNeurofeedbackPipeline:
    def test_train_decoder(self, localiser_data):
        X, y = localiser_data
        pipeline = RealtimeNeurofeedbackPipeline(
            decoder=FMRIDecoder(backend="mvpa"),
            agent=ActorCriticAgent(),
        )
        pipeline.train_decoder(X, y)
        assert pipeline._decoder_fitted

    def test_run_simulation(self, localiser_data):
        X, y = localiser_data
        pipeline = RealtimeNeurofeedbackPipeline(
            decoder=FMRIDecoder(backend="mvpa"),
            agent=ActorCriticAgent(obs_dim=2, n_actions=5),
        )
        pipeline.train_decoder(X, y)
        log = pipeline.run_simulation(n_trials=5)
        assert len(log) == 5
        assert all("similarity" in entry for entry in log)
        assert all("action" in entry for entry in log)

    def test_run_session_requires_fitted_decoder(self):
        pipeline = RealtimeNeurofeedbackPipeline()
        data_stream = iter([np.zeros(50)] * 5)
        with pytest.raises(RuntimeError):
            pipeline.run_session(data_stream, n_trials=5)

    def test_run_session_with_stream(self, localiser_data):
        X, y = localiser_data
        pipeline = RealtimeNeurofeedbackPipeline(
            decoder=FMRIDecoder(backend="mvpa"),
            agent=ActorCriticAgent(obs_dim=2, n_actions=5),
            target_class=1,
        )
        pipeline.train_decoder(X, y)

        # Stream of 5 data volumes with 50 features each
        rng = np.random.default_rng(99)
        data_stream = iter(rng.normal(size=(5, 50)))
        log = pipeline.run_session(data_stream, n_trials=5)
        assert len(log) <= 5  # may be < 5 if stream exhausted

    def test_pretrain_agent(self, localiser_data):
        X, y = localiser_data
        pipeline = RealtimeNeurofeedbackPipeline(
            decoder=FMRIDecoder(backend="mvpa"),
            agent=ActorCriticAgent(obs_dim=2, n_actions=5),
        )
        pipeline.train_decoder(X, y)
        pipeline.pretrain_agent(n_episodes=10)
        # After pretraining, agent should have episode rewards
        assert len(pipeline.agent.episode_rewards) == 10
