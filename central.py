"""One-step ActorCritic For Continuing tasks.

    * Continuing tasks
    * V function approximation.
    * Linear function approximation.
    * Full observability setting.
    * Central agent that selects actions
    for every player.

TODO:
-----
 * Add an interface for RLAgent (type agent).

References:
-----------
..[1] Sutton and Barto 2018. "Introduction to Reinforcement
  Learning 2nd Edition" (pg 333).
..[2] Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000).
  "Policy gradient methods for reinforcement learning with function
  approximation". In Advances in neural information processing
  systems (NIPS), Denver, CO. Cambridge: MIT Press.
"""

import numpy as np
from numpy.random import choice

from common import Array, Observation, Action, ActionSet, Rewards
from common import softmax


class ActorCriticCentral(object):
    """ActorCritic with Linear function approximation

    Attributes:
    ----------
    label: str
        A description for this particular agent.

    task: str
        A description for the task.

    tau: float
        The temperature parameter regulating exploration.
        Bound to interval [1, 100]

    n_players: int
        The number of players that must cover landmarks.

    n_features: int
        The number of features for the agents.

    action_set: ActionSet
        The action space, i.e, the list of all actions
        at the disposal of the agent.

    alpha: float = 0.9,
        The learning rate for policy evaluation.

    beta: float = 0.7,
        The learning rate for policy parametrization.

    explore_episodes: int = 100,
        The number of episodes to explore. The temperature
        falls linearly with the number of explore_episodes.

    explore: bool = False,
        Wheter of not the agent should take exploratory actions.

    decay: bool = False,
        Uses exponential decay on epis
        seed: int = 0,

    Methods:
    --------
    act(state): Action
        Select an action based on state

    reset(seed): None
        Resets seed, updates number of steps.

    update(state, actions, next_reward, next_state): None
        Learns from policy improvement and policy evalution.

    """
    def __init__(
        self,
        n_players: int,
        n_features: int,
        action_set: ActionSet,
        alpha: float = 0.9,
        beta: float = 0.7,
        explore_episodes: int = 100,
        explore: bool = False,
        decay: bool = False,
        seed: int = 0,
    ):

        # Attributes
        self.n_players = n_players
        self.action_set = action_set
        self.n_features = n_features

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        self.omega = np.zeros(self.n_features)
        self.theta = np.zeros((self.n_features, len(self.action_set)))
        self.zeta = 0.1
        self.mu = 0

        # Loop control
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.explore = explore
        self.decay = decay
        self.epsilon = 1.0
        self.decay_count = 1
        self.epsilon_step = float(1 / explore_episodes)
        self.reset(seed=seed)
        self.step = 0

    @property
    def label(self) -> str:
        """A description for this particular agent."""
        return "ActorCritic ({0})".format(self.task)

    @property
    def task(self) -> str:
        """Continuing or episodic."""
        return "continuing"

    @property
    def tau(self) -> float:
        """The temperature parameter regulating exploration."""
        return float(100 * self.epsilon if self.explore else 1.0)

    def reset(self, seed=None):
        """Resets seed, updates number of steps."""
        if seed is not None:
            np.random.seed(seed)

        if self.decay:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count, -0.85)
            self.beta = np.power(self.decay_count, -0.65)
        self.epsilon = float(max(1e-2, self.epsilon - self.epsilon_step))

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
        pi = self._PI(state[0])[0]
        cur = choice(len(self.action_set), p=pi)
        return self.action_set[cur]

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
        # Fully observable settings all agents see the samething.
        state = state[0]
        next_state = next_state[0]
        # Beware for two agents.
        cur = self.action_set.index(actions)

        self.delta = np.mean(next_rewards) - self.mu + (next_state - state) @ self.omega

        self.delta = np.clip(self.delta, -1, 1)
        self.mu += self.zeta * self.delta

        self.omega += self.alpha * self.delta * state
        self.omega = np.clip(self.omega, -1, 1)

        self.theta += self.beta * self.delta * self._psi(state, cur)
        self.step += 1

    def _psi(self, state: Array, action: int) -> Array:
        X = self._X(state)
        logP = self._logP(state, action)
        return (X * logP).T

    def _X(self, state: Array) -> Array:
        return np.tile(state / self.tau, (len(self.action_set), 1))

    def _logP(self, state: Array, action: int) -> Array:
        res = -np.tile(self._PI(state).T, (1, self.n_features))
        res[action, :] += 1
        return res

    def _PI(self, state: Array) -> Array:
        return softmax(self.theta.T @ state / self.tau)[None, :]


if __name__ == "__main__":
    from time import sleep

    from environment import Environment
    from plots import rewards_plot, returns_plot
    from tqdm.auto import trange

    n = 1
    seed = 0
    env = Environment(n=n, scenario="networked_spread", seed=seed, restart=True)
    agent = ActorCriticCentral(
        n_players=env.n,
        n_features=env.n_features,
        action_set=env.action_set,
        alpha=0.5,
        beta=0.3,
        explore_episodes=450,
        explore=True,
        decay=False,
        seed=seed,
    )
    first = True
    episodes = []
    rewards = []
    for episode in trange(450, desc="episodes"):
        # execution loop
        obs = env.reset()

        if not first:
            agent.reset()
        actions = agent.act(obs)

        first = False
        for _ in trange(100, desc="timesteps"):
            # step environment
            next_obs, next_rewards = env.step(actions)

            next_actions = agent.act(next_obs)

            agent.update(obs, actions, next_rewards, next_obs, next_actions)

            obs = next_obs
            actions = next_actions
            rewards.append(np.mean(next_rewards))
            episodes.append(episode)
    rewards_plot(rewards, episodes, "Train Rollouts (seed={0})".format(seed))
    returns_plot(rewards, episodes, "Train Episodes (seed={0})".format(seed))

    # This is a validation run.
    obs = env.reset()
    agent.reset()
    agent.explore = False
    actions = agent.act(obs)
    for _ in trange(100, desc="timesteps"):
        env.render()
        sleep(0.1)

        # step environment
        next_obs, next_rewards = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions

    print("Initial state x--------------------------------------------x ")
    print("[x_a, y_a, v_x, v_y, x_l, y_l] {0}".format(env.reset()[0].round(2)))
    print("x----------------------------------------------------------x ")
    print("Agents Parameters x----------------------------------------x ")
    print("agent.mu {0:0.2f}".format(agent.mu))
    print("agent.omega {0}".format(agent.omega.round(2)))
    print("agent.theta {0}".format(agent.theta.round(2)))
    print("x----------------------------------------------------------x ")
    env.reset()[0].round(2).tofile("data/initial_state.csv", sep=",")
    agent.mu.round(2).tofile("data/mu.csv", sep=",")
    agent.omega.round(2).tofile("data/omega.csv", sep=",")
    agent.theta.round(2).tofile("data/theta.csv", sep=",")
