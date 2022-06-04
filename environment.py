"""Thin wrapper around MultiAgentEnv"""
from gym import spaces
import numpy as np

from multiagent.environment import MultiAgentEnv as Base
import simple_spread as ss
import networked_spread as ns

from common import action_set, onehot, Action, Step
from consensus import consensus_matrices


class Environment(Base):
    """Environment that memoizes world and scenario for easy access.

    Attributes:
    ----------
    n_features(): int
        The number of features

    action_set(n): common.ActionSet
        The set of joint actions for all players.

    world: world
        The world (physics) and its entities (agents and landmarks.)

    action_set: common.Actions
        The set of all actions an agent can perform.

    central: bool = True
        There is a central entity that observes everything.

    communication: bool = True
        Agents may exchange information during learning.

    See Also:
    ---------
    Base.n : int
        The number of agents

    Base.action_space: gym.Discrete
        The action space.

    Base.observation_space: gym.Box
        The observation space.

    Scenario.make_world(n_players: int = 1, restart: bool = False, seed: int = 0): World
        Builds the world and its entities: players and landmarks.
    """

    @property
    def n_features(self):
        "The number of features"
        if self.central:
            return sum(map(lambda x: x.shape[0], self.observation_space))
        return self.observation_space[0].shape[0]

    def __init__(
        self,
        n: int = 1,
        scenario: str = "simple_spread",
        seed: int = 0,
        central: bool = True,
        communication: bool = False,
    ):
        if scenario not in ("simple_spread", "networked_spread"):
            raise ValueError("Invalid scenario: %s" % scenario)
        elif scenario == "simple_spread":
            self.scenario = ss.Scenario()
        else:
            self.scenario = ns.NetworkedSpreadScenario(fully_observable=central)

        self.world = self.scenario.make_world(n, seed)
        self.action_set = action_set(n)
        self.central = central
        self.communication = communication


        super(Environment, self).__init__(
            self.world,
            self.scenario.reset_world,
            self.scenario.reward,
            self.scenario.observation,
        )
        if self.central:
            obs_dim = sum([os.shape[0] for os in self.observation_space])
            self.observation_space = [
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32
                )
            ]

        if self.communication:
            self.cwms = consensus_matrices(n)

    def step(self, actions: Action) -> Step:
        next_observations, next_rewards, *_ = super(Environment, self).step(
            onehot(actions)
        )
        if self.central:
            # flatten next_observation
            next_observations = [np.concatenate([next_observations]).flatten()]

        cwm = None
        if self.communication:
            cwm = self.cwms[np.random.randint(len(self.cwms))]

        return next_observations, next_rewards, cwm

    def n_collisions(self) -> int:
        return sum([
            int(self.scenario._is_collision(i, j))
            for i, j in zip(self.world.players[:-1], self.world.players[1:])])
