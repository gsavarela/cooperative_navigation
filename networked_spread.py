""" This scenario [2] is an extension from the simple spread scenario [1].

Networked spread navigation. In this environment, `P` players must cooperate
through physical actions to reach a set of `L` landmarks. Each player is 
assigned a particular landmark to ‘cover’ (`P` = `L`). And the Reinforcement
learning player throught trial and error must discover the landmark each player
is assigned to. Furthermore, the players occupy significant physical space and
are penalized when colliding with each other. The players learn to infer the
landmark they must cover, and move there while avoiding other players

Zhang, et al. [2] implement four modifications:
    i. First, we assume the state is globally observable, i.e., the position
    of the landmarks and other players are observable to each player.
    ii. Moreover, each player has a certain target landmark to cover, and the
    individual reward is determined by the proximity to that certain landmark,
    as well as the penalty from collision with other players. In this way,
    the reward function varies between players.
    iii. The reward is further scaled by different positive coefficients,
    representing the different priority/preferences of different players.
    iv. In addition, players are connected via a time-varying communication
    network with several other players nearby.

References:
-----------
..[1] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, O. P., and Mordatch, I.
  "Multi-player actor-critic for mixed cooperative-competitive environments".
  In Advances in Neural Information Processing Systems, pp. 6382–6393, 2017

..[2] Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Basar
  2018. "Fully Decentralized Multi-player Re-inforcement Learning with
  Networked players". Proceedings of the 35th International Conference
  on Machine Learning, PMLR 80 (2018), 5872–5881.
"""
import numpy as np
from multiagent.core import World, Landmark, Agent
from multiagent.scenario import BaseScenario

from common import Observation, Array


COLORS = [(0.35, 0.35, 0.85), (0.35, 0.85, 0.35), (0.85, 0.35, 0.35)]


class NetworkedSpreadScenario(BaseScenario):
    """Cooperative navigation scenario.


    Attributes:
    ----------
    assignment(): Array
        A mapping from 'N' players to 'N' landmarks.

    coefficients(): Array
        A preference from landmarks to be covered. (set to 1)

    restart(): bool
        Change initial positions on reset. (See seed).

    seed(): int
        Regulates assignments, coefficients and entities' initial positions

    Methods:
    --------
    make_world(n_players: int = 1, restart: bool = False, seed: int = 0): World
        Builds the world and its entities: players and landmarks.

    observation(player: Agent, world: World): Observation
        Generates an observation for the Agent.

    reset_world(world: World, first: bool = False):
        Moves agents to initial position and sets velocity to zero.

    reward(player: Agent, world: World): float
        Computes the reward for player
    """
    @property
    def assignment(self) -> Array:
        """A mapping from 'N' players to 'N' landmarks."""

        return self._assignment

    @property
    def coefficients(self) -> Array:
        """A preference from landmarks to be covered. (set to 1)""" 
        return self._coefficients

    @property
    def restart(self) -> bool:
        """Change initial positions on reset. (See seed)."""
        return self._restart

    @property
    def seed(self) -> int:
        """Regulates assignments, coefficients and entities' initial positions"""
        return self._seed

    def make_world(
        self, n_players: int = 1, restart: bool = False, seed: int = 0
    ) -> World:
        """Builds the world and its entities: players and landmarks."""
        # persist decision on restart
        self._restart = restart
        self._seed = seed

        world = World()
        # set any world properties first
        # add players
        # world.agents = [Agent() for _ in range(n_players)]
        for i in range(n_players):
            agent = Agent()
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            world.agents.append(agent)
        # Ducktape agent <--> player
        world.players = world.agents

        # add landmarks
        # world.landmarks = [Landmark() for _ in range(n_players)]
        # world.landmarks = [Landmark() for _ in range(n_players)]
        # for i, landmark in enumerate(world.landmarks):
        for i in range(n_players):
            landmark = Landmark()
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            world.landmarks.append(landmark)

        # make initial conditions
        self.reset_world(world, first=True)

        return world

    def reset_world(self, world: World, first: bool = False) -> None:
        """Moves agents to initial position and sets velocity to zero.""" 
        n = len(world.players)
        # Forces initial condition to be the same.
        if first or not self.restart:
            np.random.seed(self.seed)
            self._assignment = np.random.choice(n, size=n, replace=False)
            self._coefficients = np.ones(n)

        # random properties for players
        for i, player in enumerate(world.players):
            color = COLORS[i] if i < 3 else COLORS[0]
            player.color = np.array(color)

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            j = self.assignment[i]
            color = COLORS[j] if i < 3 else (0.25, 0.25, 0.25)
            landmark.color = np.array(color)

        # set random initial states
        for player in world.players:
            player.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            player.state.p_vel = np.zeros(world.dim_p)
            player.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def _is_collision(self, player1: Agent, player2: Agent) -> bool:
        delta_pos = player1.state.p_pos - player2.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = player1.size + player2.size
        return True if dist < dist_min else False

    def reward(self, player: Agent, world: World) -> float:
        """Computes the reward for player."""
        # players are rewarded based on minimum player distance to each landmark,
        # penalized for collisions
        rew = 0
        i = world.players.index(player)
        j = self.assignment[i]
        k = self.coefficients[i]
        landmark = world.landmarks[j]
        rew = -k * np.linalg.norm(landmark.state.p_pos - player.state.p_pos)
        if player.collide:
            for a in world.players:
                if a != player and self._is_collision(a, player):
                    rew -= 1
        return rew

    def observation(self, agent: Agent, world: World) -> Observation:
        """Generates an observation for the Agent. """
        # get positions of all entities in this player's reference frame
        players_pos = [p.state.p_pos for p in world.players]
        landmarks_pos = [m.state.p_pos for m in world.landmarks]
        res = np.hstack(players_pos + landmarks_pos)
        return res


if __name__ == "__main__":
    # Those imports are to run the example.
    from time import sleep
    from multiagent.environment import MultiAgentEnv
    from common import onehot, N_ACTIONS, action_set

    action_space = action_set(1)
    # Define scenario
    scenario = NetworkedSpreadScenario()

    # Define world
    world = scenario.make_world()

    # Define multi agent environment.
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
    )

    # random action selector
    def fn():
        idx = np.random.choice(N_ACTIONS)
        res = onehot(action_space[idx])
        return res

    # execution loop
    observations = env.reset()
    for _ in range(100):
        env.render()
        sleep(0.1)

        # step environment
        obs_n, reward_n, done_n, _ = env.step(fn())
