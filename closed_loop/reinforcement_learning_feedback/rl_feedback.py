"""
rl_feedback.py
==============
Reinforcement learning agents for adaptive closed-loop neurofeedback.

Architecture
------------
NeurofeedbackEnv
    A simulation environment (Gymnasium-compatible) that models a single
    DecNef trial.  At each step the agent chooses a feedback signal level
    (discrete or continuous) and the environment returns a reward based on
    how well the participant's decoded brain state matches the target
    representation.

RLFeedbackAgent
    Policy-gradient (REINFORCE) agent with a simple MLP policy network.
    Optimises feedback signal selection to maximise cumulative reward.

ActorCriticAgent
    Actor-critic (A2C-style) agent that combines a policy network with a
    value baseline to reduce gradient variance and supports adaptive reward
    shaping based on behavioural outcomes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Simulation environment
# ---------------------------------------------------------------------------

class NeurofeedbackEnv:
    """Simulated closed-loop neurofeedback environment.

    Simulates a single DecNef session where the participant's brain state
    (decoded as a scalar similarity to the target pattern) evolves according
    to a simple drift-diffusion model modulated by feedback.

    The environment follows the Gymnasium ``step`` / ``reset`` convention but
    does not inherit from ``gymnasium.Env`` to avoid making gymnasium a hard
    dependency during unit tests.

    Parameters
    ----------
    n_steps : int
        Maximum number of feedback steps per trial (episode length).
    target_similarity : float
        Target brain-state similarity (0–1). Episode succeeds when the agent's
        decoded state exceeds this threshold.
    noise_std : float
        Standard deviation of the stochastic brain-state dynamics.
    action_space_size : int
        Number of discrete feedback levels.

    Observation space
    -----------------
    A 1-D vector ``[current_similarity, step_normalised]`` (length 2).

    Action space
    ------------
    Integer in ``[0, action_space_size)``, where higher values correspond to
    stronger positive feedback.
    """

    OBS_DIM = 2

    def __init__(
        self,
        n_steps: int = 20,
        target_similarity: float = 0.7,
        noise_std: float = 0.05,
        action_space_size: int = 5,
    ) -> None:
        self.n_steps = n_steps
        self.target_similarity = target_similarity
        self.noise_std = noise_std
        self.action_space_size = action_space_size

        self._similarity: float = 0.0
        self._step: int = 0

    # ------------------------------------------------------------------
    # Gymnasium-compatible API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment to the start of a new trial.

        Returns
        -------
        np.ndarray, shape (2,)
            Initial observation.
        """
        self._similarity = float(np.random.uniform(0.1, 0.4))
        self._step = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take one feedback step.

        Parameters
        ----------
        action : int
            Discrete feedback level in ``[0, action_space_size)``.

        Returns
        -------
        obs : np.ndarray, shape (2,)
        reward : float
        done : bool
        info : dict
        """
        # Normalise action to [0, 1]
        feedback_strength = action / max(self.action_space_size - 1, 1)

        # Brain-state drift: positive feedback nudges similarity toward target
        drift = 0.03 * feedback_strength - 0.01
        noise = np.random.normal(0, self.noise_std)
        self._similarity = float(np.clip(self._similarity + drift + noise, 0.0, 1.0))

        self._step += 1
        done = self._step >= self.n_steps or self._similarity >= self.target_similarity

        # Reward: proximity to target, bonus for reaching it
        reward = self._similarity - 0.5  # baseline reward centred at 0
        if self._similarity >= self.target_similarity:
            reward += 1.0  # success bonus

        return self._obs(), reward, done, {"similarity": self._similarity}

    def _obs(self) -> np.ndarray:
        return np.array([self._similarity, self._step / self.n_steps], dtype=np.float32)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class _PolicyNet(nn.Module):
    """MLP policy network for discrete action selection."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class _ValueNet(nn.Module):
    """MLP value network for actor-critic baseline."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# REINFORCE agent
# ---------------------------------------------------------------------------

class RLFeedbackAgent:
    """REINFORCE policy-gradient agent for adaptive neurofeedback.

    Trains a policy network to select feedback signal levels that maximise
    cumulative decoded similarity reward over a closed-loop trial.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    n_actions : int
        Number of discrete feedback actions.
    hidden_dim : int
        Policy network hidden layer size.
    lr : float
        Learning rate.
    gamma : float
        Discount factor for future rewards.
    device : str or None
        Torch device.

    Examples
    --------
    >>> env = NeurofeedbackEnv()
    >>> agent = RLFeedbackAgent(obs_dim=2, n_actions=5)
    >>> agent.train(env, n_episodes=500)
    >>> action = agent.select_action(obs)
    """

    def __init__(
        self,
        obs_dim: int = NeurofeedbackEnv.OBS_DIM,
        n_actions: int = 5,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[str] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = _PolicyNet(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.episode_rewards: list[float] = []

    def select_action(self, obs: np.ndarray) -> int:
        """Sample an action from the policy.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)

        Returns
        -------
        int
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.policy(obs_t)
        dist = torch.distributions.Categorical(probs)
        return int(dist.sample().item())

    def train(self, env: NeurofeedbackEnv, n_episodes: int = 500) -> list[float]:
        """Train the agent with the REINFORCE algorithm.

        Parameters
        ----------
        env : NeurofeedbackEnv
        n_episodes : int
            Number of training episodes.

        Returns
        -------
        list[float]
            Per-episode total rewards.
        """
        self.episode_rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            log_probs: list[torch.Tensor] = []
            rewards: list[float] = []
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                probs = self.policy(obs_t)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

                obs, reward, done, _ = env.step(int(action.item()))
                rewards.append(reward)

            total_reward = sum(rewards)
            self.episode_rewards.append(total_reward)

            # Compute discounted returns
            returns = self._compute_returns(rewards)
            returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # Policy gradient update
            loss = -(torch.stack(log_probs).squeeze() * returns_t).mean()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        return self.episode_rewards

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted cumulative returns."""
        returns: list[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns


# ---------------------------------------------------------------------------
# Actor-Critic agent
# ---------------------------------------------------------------------------

class ActorCriticAgent:
    """Actor-critic (A2C-style) agent for adaptive neurofeedback.

    Adds a learned value baseline to reduce policy gradient variance and
    supports adaptive reward shaping based on behavioural performance metrics.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    n_actions : int
        Number of discrete feedback actions.
    hidden_dim : int
        Hidden layer size for both actor and critic networks.
    lr_actor : float
        Learning rate for the policy (actor) network.
    lr_critic : float
        Learning rate for the value (critic) network.
    gamma : float
        Discount factor.
    entropy_coef : float
        Entropy regularisation coefficient (encourages exploration).
    device : str or None
        Torch device.

    Examples
    --------
    >>> env = NeurofeedbackEnv()
    >>> agent = ActorCriticAgent(obs_dim=2, n_actions=5)
    >>> agent.train(env, n_episodes=500)
    """

    def __init__(
        self,
        obs_dim: int = NeurofeedbackEnv.OBS_DIM,
        n_actions: int = 5,
        hidden_dim: int = 64,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = _PolicyNet(obs_dim, n_actions, hidden_dim).to(self.device)
        self.critic = _ValueNet(obs_dim, hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.episode_rewards: list[float] = []

    def select_action(self, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Sample an action and return its log-probability.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)

        Returns
        -------
        (action, log_prob)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.actor(obs_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def train(self, env: NeurofeedbackEnv, n_episodes: int = 500) -> list[float]:
        """Train the actor-critic agent.

        Parameters
        ----------
        env : NeurofeedbackEnv
        n_episodes : int

        Returns
        -------
        list[float]
            Per-episode total rewards.
        """
        self.episode_rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rewards: list[float] = []
            entropies: list[torch.Tensor] = []
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                probs = self.actor(obs_t)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                value = self.critic(obs_t)

                log_probs.append(dist.log_prob(action))
                values.append(value)
                entropies.append(dist.entropy())

                obs, reward, done, _ = env.step(int(action.item()))
                rewards.append(reward)

            self.episode_rewards.append(sum(rewards))

            returns = self._compute_returns(rewards)
            returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

            values_t = torch.stack(values).squeeze()
            log_probs_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies).mean()

            advantages = returns_t - values_t.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Actor loss (policy gradient with entropy bonus)
            actor_loss = -(log_probs_t * advantages).mean() - self.entropy_coef * entropy_t

            # Critic loss (mean squared TD error)
            critic_loss = F.mse_loss(values_t, returns_t)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        return self.episode_rewards

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        returns: list[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
