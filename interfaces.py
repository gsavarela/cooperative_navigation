"""Provides abstract classes that formalize a template

References
----------
..[1] https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
"""
import abc
import dill
from pathlib import Path
from typing import TypeVar

import numpy as np

import config
from common import Observation, Action, Rewards
from common import snakefy


T = TypeVar("T")


class SerializableInterface(abc.ABC):
    """Interfaces for classes that can be serialized/deserialized

    Serialize is to convert a class from memory and save in binary form.
    Conversely, deserialize loads a class from disk and instanciates to memory.

    Methods
    -------
    save_checkpoints: chkpt_dir_path: str, chkpt_sub_dir: int
        Saves the class on a path and a checkpoint number directory
    load_checkpoint: chkpt_dir_path: str, chkpt_sub_dir: int
        Deserializes class from path and checkpoint number directory
    """

    def save_checkpoints(self, chkpt_dir_path: str, chkpt_sub_dir: str):
        """Saves the class on a path and a checkpoint number directory

        Parameters
        ----------
        chkpt_dir_path: str
            The directory path that the checkpoint will be saved
        chkpt_sub_dir: str
            The sub-directory path that the checkpoint will be saved

        """
        class_name = snakefy(type(self).__name__)
        file_path = Path(chkpt_dir_path) / str(chkpt_sub_dir) / ("%s.chkpt" % class_name)
        file_path.parent.mkdir(exist_ok=True)
        with file_path.open(mode="wb") as f:
            dill.dump(self, f)

    @classmethod
    def load_checkpoint(cls, chkpt_dir_path: str, chkpt_sub_dir: str) -> T:
        """Deserializes class from path and checkpoint number directory

        Parameters
        ----------
        chkpt_dir_path: str
            The directory path that the checkpoint will be saved
        chkpt_sub_dir: str
            The sub-directory path that the checkpoint will be saved

        Returns
        -------
        obj: T
            The instance of a class cls
        """
        class_name = snakefy(cls.__name__)
        file_path = Path(chkpt_dir_path) / str(chkpt_sub_dir) / ("%s.chkpt" % class_name)
        with file_path.open(mode="rb") as f:
            new_instance = dill.load(f)

        return new_instance

class AgentInterface(SerializableInterface):
    """Multi agent system interface"""

    @property
    @abc.abstractmethod
    def fully_observable() -> bool:
        """Both rewards and the positions of every agent is known"""

    @property
    @abc.abstractmethod
    def communication() -> bool:
        """Agents may communicate during training"""

    @property
    def rng(self) -> bool:
        """Random number generator"""
        return self._rng

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
        """Resets seed, updates number of steps.

        BEWARE: Use case for seeding is object creation and
        to benchmark evaluation runs. Reseeding before an
        episode may prevent learning.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

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

