"""Simple spread scenario [1].

In this environment, agents must cooperate through physical actions to reach a
set of L landmarks. Agents observe the relative positions of other agents and
landmarks, and are collectively rewarded based on the proximity of any agent
to each landmark. In other words, the agents have to `cover` all of the
landmarks. Further, the agents occupy significant physical space and are
penalized when colliding with each other. Our agents learn to infer the
landmark they must cover, and move there while avoiding other agents.

References:
-----------
..[1] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, O. P., and Mordatch, I.
  "Multi-player actor-critic for mixed cooperative-competitive environments".
  In Advances in Neural Information Processing Systems, pp. 6382–6393, 2017

"""
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
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
    ):
        # persist decision on restart
        self._restart = restart
        self._seed = seed

        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = n_players
        num_landmarks = n_players
        # world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world, first=True)
        return world

    def reset_world(self, world: World, first: bool = False) -> None:
        """Moves agents to initial position and sets velocity to zero.""" 
        n = len(world.agents)

        if first or not self.restart:
            np.random.seed(self.seed)
            self._assignment = np.random.choice(n, size=n, replace=False)
            self._coefficients = np.ones(n)

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark,
        # penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )


if __name__ == "__main__":
    # Those imports are to run the example.
    from time import sleep
    from multiagent.environment import MultiAgentEnv
    from common import onehot, N_ACTIONS, action_set

    n = 3
    action_space = action_set(n)
    # Define scenario
    scenario = Scenario()

    # Define world
    world = scenario.make_world(n)

    # Define multi agent environment.
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
    )

    # random action selector
    def fn():
        ids = np.random.choice(N_ACTIONS, size=3, replace=True)
        actions = onehot(ids)
        return actions

    # execution loop
    observations = env.reset()
    for _ in range(100):
        env.render()
        sleep(0.1)

        # step environment
        obs_n, reward_n, done_n, _ = env.step(fn())
