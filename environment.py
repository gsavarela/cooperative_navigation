"""Thin wrapper around MultiAgentEnv"""
from multiagent.environment import MultiAgentEnv as Base
import simple_spread as ss
import networked_spread as ns
from common import action_set, ActionSet


class Environment(Base):
    """Environment that memoizes world and scenario for easy access.

    Attributes:
    ----------
    n_features(): int
        The number of features

    action_set(n): ActionSet
        The set of joint actions for all players.

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

    def __init__(
        self,
        n: int = 1,
        scenario: str = "simple_spread",
        restart: bool = False,
        seed: int = 0,
    ):
        if scenario not in ("simple_spread", "networked_spread"):
            raise ValueError("Invalid scenario: %s" % scenario)
        elif scenario == "simple_spread":
            self.scenario = ss.Scenario()
        else:
            self.scenario = ns.NetworkedSpreadScenario()

        self.world = self.scenario.make_world(n, restart, seed)
        self.action_set = action_set(n)

        super(Environment, self).__init__(
            self.world,
            self.scenario.reset_world,
            self.scenario.reward,
            self.scenario.observation,
        )

    @property
    def n_features(self):
        "The number of features"
        return sum(map(lambda x: x.shape[0], self.observation_space))
