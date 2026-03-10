"""
task.py (confidence_task)
=========================
Metacognitive confidence and decision-making task.

Implements a confidence judgement paradigm in which participants:

1. Perform a perceptual decision (e.g. orientation discrimination).
2. Rate their confidence on a continuous scale (0 = guess, 1 = certain).

The metacognitive sensitivity (meta-d') is computed to assess the degree to
which confidence ratings track decision accuracy — a key dependent measure
in RQ3 (whether neurofeedback modulates metacognitive accuracy).

Simulation mode requires no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConfidenceTrial:
    """Data class for a single confidence trial."""

    trial_id: int
    difficulty: float         # 0 (easy) → 1 (hard)
    correct: bool
    response: int             # 0 or 1
    confidence: float         # 0 = guess, 1 = certain
    rt_decision: float        # reaction time for decision (s)
    rt_confidence: float      # reaction time for confidence rating (s)
    feedback: Optional[float] = None


class ConfidenceTask:
    """Metacognitive confidence task with optional neurofeedback.

    Generates trials across a range of difficulties. Participant responses
    and confidence ratings are simulated using a signal detection theory
    framework (Gaussian signal + noise).

    Parameters
    ----------
    n_trials : int
        Total number of trials.
    difficulty_levels : int
        Number of difficulty levels (evenly spaced from easy to hard).
    dprime : float
        Sensitivity parameter (d') controlling baseline accuracy.
    with_feedback : bool
        Whether to incorporate a neurofeedback signal.
    random_state : int or None
        Random seed.

    Examples
    --------
    >>> task = ConfidenceTask(n_trials=40, dprime=1.5)
    >>> trials = task.run_simulation()
    >>> print(f"Meta-d': {task.meta_dprime():.2f}")
    """

    def __init__(
        self,
        n_trials: int = 40,
        difficulty_levels: int = 4,
        dprime: float = 1.5,
        with_feedback: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_trials = n_trials
        self.difficulty_levels = difficulty_levels
        self.dprime = dprime
        self.with_feedback = with_feedback
        self.rng = np.random.default_rng(random_state)

        self.trials: list[ConfidenceTrial] = []

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        feedback_fn: Optional[callable] = None,
    ) -> list[ConfidenceTrial]:
        """Simulate the confidence task.

        Parameters
        ----------
        feedback_fn : callable or None
            Function ``f(trial_idx) -> float`` returning a neurofeedback
            value in [0, 1].

        Returns
        -------
        list[ConfidenceTrial]
        """
        self.trials = []
        difficulties = np.linspace(0.1, 1.0, self.difficulty_levels)
        diff_seq = self.rng.choice(difficulties, size=self.n_trials)

        for trial_id, difficulty in enumerate(diff_seq):
            # Effective d' decreases as difficulty increases
            effective_dprime = self.dprime * (1.0 - 0.7 * difficulty)

            # Decision: signal detection theory
            signal_strength = self.rng.normal(effective_dprime / 2, 1.0)
            response = int(signal_strength > 0.0)
            correct = response == 1  # signal is always class 1 in simulation

            # Confidence: proportional to |signal_strength| (type-2 SDT)
            confidence = float(np.clip(np.abs(signal_strength) / (self.dprime + 1e-6), 0.0, 1.0))

            # Simulated reaction times
            rt_decision = float(self.rng.gamma(shape=3.0, scale=0.15))
            rt_confidence = float(self.rng.gamma(shape=3.0, scale=0.2))

            feedback = None
            if self.with_feedback and feedback_fn is not None:
                feedback = float(feedback_fn(trial_id))

            self.trials.append(
                ConfidenceTrial(
                    trial_id=trial_id,
                    difficulty=float(difficulty),
                    correct=correct,
                    response=response,
                    confidence=confidence,
                    rt_decision=rt_decision,
                    rt_confidence=rt_confidence,
                    feedback=feedback,
                )
            )

        return self.trials

    # ------------------------------------------------------------------
    # Metacognitive metrics
    # ------------------------------------------------------------------

    def accuracy(self) -> float:
        """Return mean accuracy."""
        if not self.trials:
            return float("nan")
        return float(np.mean([t.correct for t in self.trials]))

    def mean_confidence(self) -> float:
        """Return mean confidence."""
        if not self.trials:
            return float("nan")
        return float(np.mean([t.confidence for t in self.trials]))

    def phi_coefficient(self) -> float:
        """Return phi correlation between correctness and high confidence.

        A positive phi indicates that confidence tracks accuracy (a necessary
        condition for good metacognition).
        """
        if not self.trials:
            return float("nan")

        correct = np.array([t.correct for t in self.trials], dtype=float)
        high_conf = np.array([t.confidence >= 0.5 for t in self.trials], dtype=float)

        # Phi = Pearson r for two binary variables
        if correct.std() == 0 or high_conf.std() == 0:
            return 0.0
        return float(np.corrcoef(correct, high_conf)[0, 1])

    def meta_dprime(self) -> float:
        """Return a simplified meta-d' estimate (type-2 sensitivity).

        Uses the ratio of confidence-weighted hit rate to false alarm rate
        as a proxy for full meta-d' (which requires fitting signal detection
        models to confidence rating data).
        """
        hits = [t.confidence for t in self.trials if t.correct]
        fas = [t.confidence for t in self.trials if not t.correct]
        if not hits or not fas:
            return float("nan")
        mean_hit_conf = np.mean(hits)
        mean_fa_conf = np.mean(fas)
        # Simplified: log odds ratio as proxy
        eps = 1e-6
        return float(np.log((mean_hit_conf + eps) / (mean_fa_conf + eps)))

    def report(self) -> None:
        """Print a summary of metacognitive performance."""
        print(
            f"ConfidenceTask | Trials: {len(self.trials)} | "
            f"Accuracy: {self.accuracy():.2%} | "
            f"Mean Confidence: {self.mean_confidence():.2f} | "
            f"Phi: {self.phi_coefficient():.3f} | "
            f"Meta-d' (proxy): {self.meta_dprime():.3f}"
        )
