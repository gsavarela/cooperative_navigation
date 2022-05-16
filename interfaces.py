"""Provides abstract classes that formalize a template"""
import abc
from common import Observation, Action, Rewards


class AgentInterface(abc.ABC):
    """Multi agent system interface"""

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

    @abc.abstractmethod
    def reset(self, seed: int = None) -> None:
        """Resets seed and updates."""
