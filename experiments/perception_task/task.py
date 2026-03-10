"""
task.py (perception_task)
=========================
Visual discrimination task for closed-loop neurofeedback studies.

Implements a two-alternative forced-choice (2AFC) visual discrimination
paradigm.  On each trial:

1. A fixation cross is shown.
2. A stimulus (grating / noise patch / face image) is briefly presented.
3. The participant responds (keyboard / button box).
4. Feedback is optionally delivered (auditory tone / visual indicator).
5. Metacognitive confidence is recorded (if ``with_confidence=True``).

The task can operate:

* In **simulation mode** (no display): generates synthetic trial sequences for
  pipeline testing without any hardware.
* In **real mode**: requires PsychoPy for stimulus presentation (optional dep).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrialResult:
    """Data class for storing per-trial results."""

    trial_id: int
    stimulus: str
    response: Optional[int]          # 0 or 1 (2AFC choice)
    correct: Optional[bool]
    rt: Optional[float]              # reaction time (s)
    confidence: Optional[float]      # 0–1 scale
    feedback: Optional[float]        # feedback signal delivered
    onset_time: float = 0.0


class PerceptionTask:
    """Two-alternative forced-choice visual discrimination task.

    Parameters
    ----------
    n_trials : int
        Number of trials per block.
    stimulus_duration : float
        Duration of the stimulus presentation (seconds).
    isi_duration : float
        Inter-stimulus interval duration (seconds).
    with_confidence : bool
        Whether to collect a confidence rating after each response.
    with_feedback : bool
        Whether to deliver neurofeedback after each trial.
    random_state : int or None
        Random seed for reproducibility in simulation mode.

    Examples
    --------
    >>> task = PerceptionTask(n_trials=20, with_feedback=True)
    >>> results = task.run_simulation(feedback_fn=lambda obs: 0.5)
    >>> task.report()
    """

    STIMULI = ["stimulus_A", "stimulus_B"]

    def __init__(
        self,
        n_trials: int = 20,
        stimulus_duration: float = 0.2,
        isi_duration: float = 1.5,
        with_confidence: bool = True,
        with_feedback: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_trials = n_trials
        self.stimulus_duration = stimulus_duration
        self.isi_duration = isi_duration
        self.with_confidence = with_confidence
        self.with_feedback = with_feedback
        self.rng = np.random.default_rng(random_state)

        self.results: list[TrialResult] = []

    # ------------------------------------------------------------------
    # Simulation mode (no display hardware required)
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        feedback_fn: Optional[callable] = None,
    ) -> list[TrialResult]:
        """Run a fully simulated perception task.

        Parameters
        ----------
        feedback_fn : callable or None
            Function ``f(trial_idx) -> float`` that returns a feedback
            value in [0, 1] for each trial.  When *None*, feedback is
            generated randomly.

        Returns
        -------
        list[TrialResult]
        """
        self.results = []
        stimuli_seq = self.rng.choice(self.STIMULI, size=self.n_trials)
        correct_responses = {"stimulus_A": 0, "stimulus_B": 1}

        for trial_id, stimulus in enumerate(stimuli_seq):
            onset_time = float(time.monotonic())

            # Simulate participant response (70 % accuracy, noisy RT)
            correct_response = correct_responses[stimulus]
            is_correct = self.rng.random() < 0.7
            response = correct_response if is_correct else 1 - correct_response
            rt = float(self.rng.normal(0.5, 0.1))

            # Optional confidence rating (0–1, correlated with accuracy)
            confidence = None
            if self.with_confidence:
                base = 0.75 if is_correct else 0.35
                confidence = float(np.clip(self.rng.normal(base, 0.15), 0.0, 1.0))

            # Optional feedback signal
            feedback_value = None
            if self.with_feedback:
                feedback_value = (
                    float(feedback_fn(trial_id))
                    if feedback_fn is not None
                    else float(self.rng.uniform(0.0, 1.0))
                )

            result = TrialResult(
                trial_id=trial_id,
                stimulus=stimulus,
                response=response,
                correct=bool(is_correct),
                rt=rt,
                confidence=confidence,
                feedback=feedback_value,
                onset_time=onset_time,
            )
            self.results.append(result)

        return self.results

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def accuracy(self) -> float:
        """Return mean accuracy across completed trials."""
        correct_trials = [r for r in self.results if r.correct is not None]
        if not correct_trials:
            return float("nan")
        return float(np.mean([r.correct for r in correct_trials]))

    def mean_rt(self) -> float:
        """Return mean reaction time (seconds)."""
        rts = [r.rt for r in self.results if r.rt is not None]
        if not rts:
            return float("nan")
        return float(np.mean(rts))

    def mean_confidence(self) -> float:
        """Return mean confidence rating."""
        confs = [r.confidence for r in self.results if r.confidence is not None]
        if not confs:
            return float("nan")
        return float(np.mean(confs))

    def report(self) -> None:
        """Print a summary of task performance."""
        print(
            f"PerceptionTask | Trials: {len(self.results)} | "
            f"Accuracy: {self.accuracy():.2%} | "
            f"Mean RT: {self.mean_rt():.3f}s | "
            f"Mean Confidence: {self.mean_confidence():.2f}"
        )
