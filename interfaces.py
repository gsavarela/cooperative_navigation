"""Provides abstract classes that formalize a template"""
import abc

import numpy as np
from common import Observation, Action, Rewards
import config


class AgentInterface(abc.ABC):
    """Multi agent system interface"""

    @property
    @abc.abstractmethod
    def fully_observable() -> bool:
        """Both rewards and the positions of every agent is known"""

    @property
    @abc.abstractmethod
    def communication() -> bool:
        """Agents may communicate during training"""

    @abc.abstractmethod
    def act(self, state: Observation) -> Action:
        """Select an action based on state

        Parameters:
        -----------
        state: Observation
            The state of the world according to the player.

        Returns:
        --------
        actions: Actions
            The actions for all players.

        """

    @abc.abstractmethod
    def update(
        self,
        state: Observation,
        actions: Action,
        next_rewards: Rewards,
        next_state: Observation,
        next_actions: Action,
    ) -> None:
        """Learns from policy improvement and policy evalution.

        Parameters:
        -----------
        state: Observation
            The state of the world according to the player, at t.

        actions: Action
            The action for every player, selected at t.

        next_rewards: Rewards
            The reward earned for each player.

        next_state: Observation
            The state of the world according to the player, at t + 1.

        next_actions: Action
            The action for every player, select at t + 1.
        """

    def reset(self, seed: int = None) -> None:
        """Resets seed, updates number of steps."""
        if seed is not None:
            np.random.seed(seed)

        if self.decay:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count, -0.85)
            self.beta = np.power(self.decay_count, -0.65)
        self.episodes += 1


class ActorCriticInterface(abc.ABC):
    """Actor Critic with Linear function approximation"""

    @property
    def label(self) -> str:
        """A description for this particular agent."""
        return "{0} ({1})".format(self.__class__.__name__, self.task)

    @property
    def task(self) -> str:
        """Continuing or episodic."""
        return "continuing"

    @property
    def tau(self) -> float:
        """The temperature parameter regulating exploration."""
        if self.explore:
            return max(100 - (self.episodes - 1) * self.epsilon_step, config.TAU)
        else:
            return config.TAU
