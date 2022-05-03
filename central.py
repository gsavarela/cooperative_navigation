"""One-step ActorCritic For Continuing tasks.

    * Continuing tasks
    * V function approximation.
    * Linear function approximation.
    * Full observability setting.
    * Central agent that selects actions for every player.

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
from typing import List

import numpy as np
from numpy.random import choice

import config
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
        self.epsilon_step = float(1 / (config.TAU * explore_episodes))
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
        self.epsilon = float(max(config.TAU * 1e-2, self.epsilon - self.epsilon_step))

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
        # self.delta = np.clip(self.delta, -1, 1)

        self.mu += self.zeta * self.delta
        self.omega += self.alpha * self.delta * state
        # self.omega = np.clip(self.omega, -1, 1)

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
    from pathlib import Path
    import common

    import pandas as pd
    from environment import Environment

    from plots import save_frames_as_gif
    from plots import metrics_plot, returns_plot
    from tqdm.auto import trange
    from numpy.linalg import norm

    # Some helpful functions
    def plot_rewards(rewards, episodes) -> None:
        metrics_plot(
            rewards,
            episodes,
            "Average Rewards",
            "Train Rollouts (seed={0})".format(seed),
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        )

    def plot_mus(mus, episodes) -> None:
        metrics_plot(
            mus,
            episodes,
            "mu",
            "Train Mu (seed={0})".format(seed),
            rollouts=False,
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        )

    def plot_omegas(omegas, episodes) -> None:
        metrics_plot(
            omegas,
            episodes,
            "|omega|",
            "Train omega (seed={0})".format(seed),
            rollouts=False,
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        )

    def plot_returns(rewards, episodes) -> None:
        returns_plot(
            rewards,
            episodes,
            "Average Rewards",
            "Train Returns (seed={0})".format(seed),
            save_directory_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        )

    def save(data, filename) -> None:
        path = Path(config.BASE_PATH) / "{0:02d}".format(config.SEED)
        csvname = "{0}.csv".format(filename)
        file_path = path / csvname
        pd.DataFrame(data=data.round(2)).to_csv(file_path.as_posix(), sep=",")

    def critic(
        deltas: List[float],
        rewards: List[float],
        mus: List[float],
        dxs: List[float],
        omegas0: List[float],
        omegas1: List[float],
        xs: List[float],
    ) -> None:
        path = Path(config.BASE_PATH) / "{0:02d}".format(config.SEED)
        # Create individual dataframes
        deltas_df = pd.DataFrame(np.vstack(deltas), columns=["deltas"])
        rewards_df = pd.DataFrame(np.vstack(rewards), columns=["rewards"])
        mus_df = pd.DataFrame(np.vstack(mus), columns=["mus"])
        dxs_df = pd.DataFrame(
            np.vstack(dxs), columns=["dx_a", "dy_a", "dv_x", "dv_y", "dx_r", "dy_r"]
        )
        omegas0_df = pd.DataFrame(
            np.vstack(omegas0), columns=["w0_1", "w0_2", "w0_3", "w0_4", "w0_5", "w0_6"]
        )
        omegas1_df = pd.DataFrame(
            np.vstack(omegas1), columns=["w1_1", "w1_2", "w1_3", "w1_4", "w1_5", "w1_6"]
        )
        xs_df = pd.DataFrame(
            np.vstack(xs), columns=["x_a", "y_a", "v_x", "d_y", "x_r", "y_r"]
        )

        # 1. delta dataframe
        dataframes = [deltas_df, rewards_df, mus_df, dxs_df, omegas0_df]
        df = pd.concat(dataframes, axis=1)
        df.to_csv(path / "deltas-seed{0:02d}.csv".format(config.SEED))

        # 2. mu dataframe
        dataframes = [mus_df, deltas_df]
        df = pd.concat(dataframes, axis=1)
        df.to_csv(path / "mus-seed{0:02d}.csv".format(config.SEED))

        # 3. omegas dataframe
        dataframes = [omegas1_df, omegas0_df, deltas_df, xs_df]
        df = pd.concat(dataframes, axis=1)
        df.to_csv(path / "omegas-seed{0:02d}.csv".format(config.SEED))

    def traces(
        x0: List[Array],
        actions: List[int],
        x1: List[Array],
        vs: List[float],
        pis: List[Array],
    ) -> None:

        path = Path(config.BASE_PATH) / "{0:02d}".format(config.SEED)
        actions = [str(common.PlayerActions(act).name) for act in actions]

        # individual dataframes
        x0_df = pd.DataFrame(
            data=np.vstack(x0), columns=["x0_1", "x0_2", "x0_3", "x0_4", "x0_5", "x0_6"]
        )
        actions_df = pd.DataFrame(data=actions, columns=["actions"])
        x1_df = pd.DataFrame(
            data=np.vstack(x1), columns=["x1_1", "x1_2", "x1_3", "x1_4", "x1_5", "x1_6"]
        )

        vs_df = pd.DataFrame(data=np.vstack(vs), columns=["v(x)"])

        pis_df = pd.DataFrame(
            data=np.vstack(pis),
            columns=[str(common.PlayerActions(act).name) for act in range(5)],
        )

        # delta dataframe
        dataframes = [x0_df, actions_df, x1_df, vs_df, pis_df]
        df = pd.concat(dataframes, axis=1)
        df.to_csv(path / "traces-seed{0:02d}.csv".format(config.SEED))

    n = 1
    seed = config.SEED
    env = Environment(n=n, scenario="networked_spread", seed=seed, restart=True)
    agent = ActorCriticCentral(
        n_players=env.n,
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
    omegas0 = []
    omegas1 = []
    thetas = []
    dxs = []
    deltas = []
    xs = []
    acts = []
    xs1 = []
    vs = []
    pis = []
    for episode in trange(config.EXPLORE_EPISODES, desc="episodes"):
        # execution loop
        obs = env.reset()

        if not first:
            agent.reset()
        actions = agent.act(obs)

        first = False
        for _ in trange(100, desc="timesteps"):

            # step environment
            next_obs, next_rewards = env.step(actions)

            dxs.append((next_obs[0] - obs[0]).copy())
            xs.append(obs[0].copy())
            mus.append(agent.mu)
            omegas0.append(agent.omega.copy())
            acts.append(actions[0])
            vs.append(obs[0] @ agent.omega)
            pis.append(agent._PI(obs[0]))

            next_actions = agent.act(next_obs)

            agent.update(obs, actions, next_rewards, next_obs, next_actions)

            obs = next_obs
            actions = next_actions

            rewards.append(np.mean(next_rewards))
            episodes.append(episode)
            omegas1.append(agent.omega.copy())
            thetas.append(agent.theta.flatten())
            deltas.append(agent.delta)
            xs1.append(obs[0].copy())

    plot_rewards(rewards, episodes)
    plot_mus(mus, episodes)
    plot_omegas([norm(omg) for omg in omegas1], episodes)
    # This is a validation run.
    obs = env.reset()
    agent.reset()
    agent.explore = False
    actions = agent.act(obs)
    frames = []
    for _ in trange(100, desc="timesteps"):
        # env.render()  # for humans
        sleep(0.1)
        frames += env.render(mode="rgb_array")  # for saving

        # step environment
        next_obs, next_rewards = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions

    critic(deltas, rewards, mus, dxs, omegas0, omegas1, xs)
    traces(xs, acts, xs1, vs, pis)
    save(np.vstack(thetas), "theta_seed{0:02d}".format(config.SEED))
    save_frames_as_gif(
        frames,
        dir_path=Path(config.BASE_PATH) / "{0:02d}".format(config.SEED),
        filename="simulation-seed{0:02d}.gif".format(seed),
    )
