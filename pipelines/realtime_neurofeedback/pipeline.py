"""
pipeline.py
===========
Real-time closed-loop neurofeedback pipeline.

Architecture
------------
The pipeline integrates three components:

1. **Neural decoder** (fMRI / EEG / fNIRS) – decodes the participant's current
   brain state into a similarity score relative to the target neural pattern.

2. **Cross-modal mapper** (optional) – projects EEG/fNIRS features into the
   shared fMRI embedding space so that accessible modalities can approximate
   decoded fMRI states.

3. **RL feedback agent** – selects a feedback signal level based on the current
   decoded state, optimising toward the target representation over the course
   of the session.

The pipeline is designed to be modular: each component can be replaced
independently and the pipeline can run in simulation mode (using
``NeurofeedbackEnv``) or connected to a real neuroimaging system.

Usage
-----
>>> pipeline = RealtimeNeurofeedbackPipeline(
...     decoder=FMRIDecoder(backend="mvpa"),
...     agent=ActorCriticAgent(obs_dim=2, n_actions=5),
... )
>>> pipeline.train_decoder(X_train, y_train)
>>> results = pipeline.run_session(data_stream)
"""

from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

import numpy as np

from closed_loop.reinforcement_learning_feedback.rl_feedback import (
    ActorCriticAgent,
    NeurofeedbackEnv,
    RLFeedbackAgent,
)
from models.fmri_decoder.fmri_decoder import FMRIDecoder


class RealtimeNeurofeedbackPipeline:
    """End-to-end real-time neurofeedback pipeline.

    Parameters
    ----------
    decoder : FMRIDecoder or any object with ``predict_proba(X)`` method
        Neural state decoder.
    agent : RLFeedbackAgent or ActorCriticAgent
        RL agent that selects feedback levels.
    cross_modal_mapper : object with ``transform(X_fmri, X_eeg)`` method, optional
        Cross-modal mapper to project EEG/fNIRS into the fMRI space.
    target_class : int
        Target class whose predicted probability is used as the decoded
        similarity score fed to the RL agent.
    feedback_delay : float
        Delay (seconds) between receiving a brain volume and delivering
        feedback. Set to 0 for simulation.
    verbose : bool
        Print step-by-step information.

    Examples
    --------
    >>> pipeline = RealtimeNeurofeedbackPipeline(
    ...     decoder=FMRIDecoder(backend="mvpa"),
    ...     agent=ActorCriticAgent(),
    ... )
    >>> pipeline.train_decoder(X_train, y_train)
    >>> pipeline.pretrain_agent(n_episodes=200)
    >>> results = pipeline.run_session(data_stream, n_trials=10)
    """

    def __init__(
        self,
        decoder: Optional[FMRIDecoder] = None,
        agent: Optional[RLFeedbackAgent | ActorCriticAgent] = None,
        cross_modal_mapper=None,
        target_class: int = 1,
        feedback_delay: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.decoder = decoder or FMRIDecoder(backend="mvpa")
        self.agent = agent or ActorCriticAgent()
        self.cross_modal_mapper = cross_modal_mapper
        self.target_class = target_class
        self.feedback_delay = feedback_delay
        self.verbose = verbose

        self._decoder_fitted = False
        self.session_log: list[dict] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_decoder(self, X: np.ndarray, y: np.ndarray) -> "RealtimeNeurofeedbackPipeline":
        """Train the neural state decoder offline.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Neural patterns (voxels / features) from the localiser run.
        y : np.ndarray, shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        self.decoder.fit(X, y)
        self._decoder_fitted = True
        if self.verbose:
            print("[Pipeline] Decoder training complete.")
        return self

    def pretrain_agent(
        self,
        n_episodes: int = 500,
        env: Optional[NeurofeedbackEnv] = None,
    ) -> "RealtimeNeurofeedbackPipeline":
        """Pre-train the RL agent in a simulated environment.

        Parameters
        ----------
        n_episodes : int
            Number of simulated episodes.
        env : NeurofeedbackEnv or None
            Custom simulation environment. Defaults to a standard environment.

        Returns
        -------
        self
        """
        env = env or NeurofeedbackEnv()
        self.agent.train(env, n_episodes=n_episodes)
        if self.verbose:
            print(f"[Pipeline] Agent pre-training complete ({n_episodes} episodes).")
        return self

    # ------------------------------------------------------------------
    # Real-time session
    # ------------------------------------------------------------------

    def run_session(
        self,
        data_stream: Iterator[np.ndarray],
        n_trials: int = 20,
    ) -> list[dict]:
        """Run a closed-loop neurofeedback session.

        Parameters
        ----------
        data_stream : Iterator[np.ndarray]
            Yields neural data volumes (shape ``(n_features,)`` per volume).
        n_trials : int
            Number of feedback trials.

        Returns
        -------
        list[dict]
            Session log with per-trial decoded states, actions, and rewards.
        """
        if not self._decoder_fitted:
            raise RuntimeError(
                "Decoder must be fitted before running a session. Call train_decoder() first."
            )

        self.session_log = []
        obs = np.array([0.2, 0.0], dtype=np.float32)  # initial observation

        for trial in range(n_trials):
            # Acquire neural data for this trial
            try:
                data = next(data_stream)
            except StopIteration:
                if self.verbose:
                    print(f"[Pipeline] Data stream exhausted at trial {trial}.")
                break

            # Decode brain state → similarity score
            proba = self.decoder.predict_proba(data.reshape(1, -1))
            similarity = float(proba[0, self.target_class])

            # Build observation for RL agent
            obs = np.array([similarity, trial / n_trials], dtype=np.float32)

            # Select feedback action
            if isinstance(self.agent, ActorCriticAgent):
                action, _ = self.agent.select_action(obs)
            else:
                action = self.agent.select_action(obs)

            # Optionally delay (real hardware)
            if self.feedback_delay > 0:
                time.sleep(self.feedback_delay)

            # Compute reward signal (proximity to target)
            reward = similarity - 0.5

            log_entry = {
                "trial": trial,
                "similarity": similarity,
                "action": action,
                "reward": reward,
            }
            self.session_log.append(log_entry)

            if self.verbose:
                print(
                    f"[Trial {trial:03d}] similarity={similarity:.3f} "
                    f"action={action} reward={reward:+.3f}"
                )

        return self.session_log

    # ------------------------------------------------------------------
    # Simulation mode
    # ------------------------------------------------------------------

    def run_simulation(self, n_trials: int = 20) -> list[dict]:
        """Run a fully simulated closed-loop session (no real data required).

        Parameters
        ----------
        n_trials : int
            Number of simulated feedback trials.

        Returns
        -------
        list[dict]
            Session log.
        """
        env = NeurofeedbackEnv(n_steps=n_trials)
        obs = env.reset()
        self.session_log = []

        for trial in range(n_trials):
            if isinstance(self.agent, ActorCriticAgent):
                action, _ = self.agent.select_action(obs)
            else:
                action = self.agent.select_action(obs)

            obs, reward, done, info = env.step(action)

            log_entry = {
                "trial": trial,
                "similarity": info["similarity"],
                "action": action,
                "reward": reward,
            }
            self.session_log.append(log_entry)

            if self.verbose:
                print(
                    f"[Sim Trial {trial:03d}] similarity={info['similarity']:.3f} "
                    f"action={action} reward={reward:+.3f}"
                )

            if done:
                break

        return self.session_log
