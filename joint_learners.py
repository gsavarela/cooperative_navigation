"""One-step ActorCritic For Continuing tasks.

    * Continuing tasks
    * V function approximation.
    * Linear function approximation.
    * Fully observable setting.
    * Same reward for both agents.
    * Fully cooperative setting.

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
from pathlib import Path

import numpy as np
from numpy.random import choice

import config
from common import Array, Observation, Action, ActionSet, Rewards
from common import softmax


class ActorCriticJoint(object):
    """Joint actor critic with Linear function approximation

    Centralized critic and independent actors.

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
        self.n_actions = int(np.power(len(action_set), 1 / n_players))
        self.n_features = n_features

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        self.omega = np.zeros(self.n_features)
        self.theta = np.zeros((self.n_players, self.n_features, self.n_actions))
        self.zeta = 0.1
        self.mu = 0

        # Loop control
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.explore = explore
        self.decay = decay
        self.epsilon = 1.0
        self.decay_count = 1
        self.epsilon_step = float((100 - config.TAU) / explore_episodes)

        self.step = 0
        self.episodes = 0
        self.reset(seed=seed)

    @property
    def label(self) -> str:
        """A description for this particular agent."""
        return "ActorCriticJoint(N={0})".format(self.n_players)

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

    def reset(self, seed=None):
        """Resets seed, updates number of steps."""
        if seed is not None:
            np.random.seed(seed)

        if self.decay:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count, -0.85)
            self.beta = np.power(self.decay_count, -0.65)
        self.episodes += 1

    def act(self, state: Observation) -> Action:
        """Select an action based on state

        Parameters
        ----------
        state: Observation
            The state of the world according to the player.

        Returns
        -------
        actions: Actions
            The actions for all players.
        """
        ret = []
        for n in range(self.n_players):
            pi = self._PI(np.hstack(state), n)
            ret.append(choice(5, p=pi.flatten()))
        return tuple(ret)

    def update(
        self,
        state: Observation,
        actions: Action,
        next_rewards: Rewards,
        next_state: Observation,
        next_actions: Action,
    ) -> None:
        """Learns from policy improvement and policy evalution.

        Parameters
        ----------
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
        _s0 = np.hstack(state)
        _s1 = np.hstack(next_state)

        # Beware for two agents.
        delta = (
            np.mean(next_rewards)
            - self.mu
            + (_s1 - _s0) @ self.omega
        )
        delta = np.clip(delta, -1, 1)
        self.omega += self.alpha * delta * _s0
        self.omega = np.clip(self.omega, -1, 1)

        for n in range(self.n_players):
            self.theta[n] += (
                self.beta * delta * self._psi(_s0, actions[n], n)
            )

        self.mu += self.zeta * delta
        self.step += 1

    def _psi(self, state: Array, action: int, n: int) -> Array:
        _X = self._X(state)
        _logP = self._logP(state, action, n)
        return (_X * _logP).T

    def _X(self, state: Array) -> Array:
        return np.tile(state / self.tau, (self.n_actions, 1))

    def _logP(self, state: Array, action: int, n: int) -> Array:
        res = -np.tile(self._PI(state, n).T, (1, self.n_features))
        res[action, :] += 1
        return res

    def _PI(self, state: Array, n: int) -> Array:
        ret = softmax(self.theta[n, :].T @ state / self.tau)[None, :]
        return ret

    def _pi(self, state: Array, n: int) -> Array:
        ret = softmax(self.theta[n, :].T @ state)[None, :]
        return ret


if __name__ == "__main__":
    from time import sleep
    from pathlib import Path
    from environment import Environment
    import pandas as pd

    from plots import save_frames_as_gif
    from plots import metrics_plot, returns_plot
    from tqdm.auto import trange

    # Some helpful functions
    def plot_rewards(rewards, episodes) -> None:
        metrics_plot(
            rewards,
            "Average Rewards",
            "Train Rollouts (seed={0})".format(seed),
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
            episodes=episodes,
        )

    def plot_mus(mus, episodes) -> None:
        metrics_plot(
            mus,
            "mu",
            "Train Mu (seed={0})".format(seed),
            rollouts=False,
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
            episodes=episodes,
        )

    def plot_returns(rewards, episodes) -> None:
        returns_plot(
            rewards,
            episodes,
            "Train Returns (seed={0})".format(seed),
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        )

    def plot_eval(rewards) -> None:
        metrics_plot(
            rewards,
            "Average Rewards",
            "Evaluation Rollouts (seed={0})".format(seed),
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        )


    def save(data, filename) -> None:
        path = Path(config.BASE_PATH) / "{0:02d}".format(config.SEED)
        csvname = "{0}.csv".format(filename)
        file_path = path / csvname
        pd.DataFrame(data=data.round(2)).to_csv(file_path.as_posix(), sep=",")

    seed = config.SEED
    env = Environment(
        n=config.N_AGENTS,
        scenario="networked_spread",
        seed=seed,
        central=True,
    )
    agent = ActorCriticJoint(
        n_players=config.N_AGENTS,
        n_features=env.n_features,
        action_set=env.action_set,
        alpha=config.ALPHA,
        beta=config.BETA,
        explore_episodes=config.EXPLORE_EPISODES,
        explore=config.EXPLORE,
        decay=False,
        seed=config.SEED,
    )
    first = True
    episodes = []
    rewards = []
    mus = []
    for episode in trange(config.EPISODES, desc="episodes"):
        # execution loop
        obs = env.reset()

        if not first:
            agent.reset()
        actions = agent.act(obs)

        first = False
        for _ in trange(100, desc="timesteps"):

            # step environment
            next_obs, next_rewards = env.step(actions)

            # actor parameters.
            next_actions = agent.act(next_obs)

            agent.update(obs, actions, next_rewards, next_obs, next_actions)

            obs = next_obs
            actions = next_actions

            rewards.append(np.mean(next_rewards))
            episodes.append(episode)
            mus.append(np.mean(agent.mu))

    plot_returns(rewards, episodes)
    plot_rewards(rewards, episodes)
    plot_mus(mus, episodes)
    # This is a validation run.
    obs = env.reset()
    agent.reset()
    agent.explore = False
    actions = agent.act(obs)
    frames = []
    rewards = []
    for _ in trange(100, desc="timesteps"):
        # env.render()  # for humans
        sleep(0.1)
        frames += env.render(mode="rgb_array")  # for saving

        # step environment
        next_obs, next_rewards = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions

        rewards.append(np.mean(next_rewards))

    plot_eval(rewards)
    save_frames_as_gif(
        frames,
        dir_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        filename="simulation-seed{0:02d}.gif".format(seed),
    )
