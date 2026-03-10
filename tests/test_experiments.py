"""
tests/test_experiments.py
=========================
Unit tests for the experimental task modules.
"""

import numpy as np
import pytest

from experiments.confidence_task import ConfidenceTask
from experiments.perception_task import PerceptionTask


class TestPerceptionTask:
    def test_run_simulation_returns_results(self):
        task = PerceptionTask(n_trials=20, random_state=42)
        results = task.run_simulation()
        assert len(results) == 20

    def test_accuracy_in_range(self):
        task = PerceptionTask(n_trials=50, random_state=0)
        task.run_simulation()
        acc = task.accuracy()
        assert 0.0 <= acc <= 1.0

    def test_with_confidence(self):
        task = PerceptionTask(n_trials=10, with_confidence=True, random_state=1)
        results = task.run_simulation()
        assert all(r.confidence is not None for r in results)

    def test_without_confidence(self):
        task = PerceptionTask(n_trials=10, with_confidence=False, random_state=2)
        results = task.run_simulation()
        assert all(r.confidence is None for r in results)

    def test_with_feedback_fn(self):
        task = PerceptionTask(n_trials=10, with_feedback=True, random_state=3)
        results = task.run_simulation(feedback_fn=lambda t: 0.75)
        assert all(r.feedback == pytest.approx(0.75) for r in results)

    def test_mean_rt_positive(self):
        task = PerceptionTask(n_trials=30, random_state=4)
        task.run_simulation()
        assert task.mean_rt() > 0.0


class TestConfidenceTask:
    def test_run_simulation_returns_trials(self):
        task = ConfidenceTask(n_trials=30, random_state=42)
        trials = task.run_simulation()
        assert len(trials) == 30

    def test_accuracy_in_range(self):
        task = ConfidenceTask(n_trials=40, dprime=2.0, random_state=0)
        task.run_simulation()
        assert 0.0 <= task.accuracy() <= 1.0

    def test_confidence_in_range(self):
        task = ConfidenceTask(n_trials=30, random_state=1)
        task.run_simulation()
        assert 0.0 <= task.mean_confidence() <= 1.0

    def test_phi_coefficient(self):
        task = ConfidenceTask(n_trials=100, dprime=2.5, random_state=5)
        task.run_simulation()
        phi = task.phi_coefficient()
        assert -1.0 <= phi <= 1.0

    def test_meta_dprime(self):
        task = ConfidenceTask(n_trials=50, dprime=2.0, random_state=6)
        task.run_simulation()
        meta_d = task.meta_dprime()
        assert not np.isnan(meta_d)

    def test_with_feedback(self):
        task = ConfidenceTask(n_trials=10, with_feedback=True, random_state=7)
        trials = task.run_simulation(feedback_fn=lambda t: 0.6)
        assert all(tr.feedback == pytest.approx(0.6) for tr in trials)
