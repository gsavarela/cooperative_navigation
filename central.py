"""One-step ActorCritic For Continuing tasks.

    * Continuing tasks
    * V function approximation.
    * Linear function approximation.
    * Full observability setting.
    * Central agent that selects actions for every player.

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

import config
from common import Array, Observation, Action, ActionSet, Rewards
from common import softmax, make_dirs
from interfaces import AgentInterface, ActorCriticInterface


class ActorCriticCentral(AgentInterface, ActorCriticInterface):
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

    zeta: float = 0.1,
        The weighting on the global average.

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

    fully_observable = True
    communication = False

    def __init__(
        self,
        n_players: int,
        n_features: int,
        action_set: ActionSet,
        alpha: float = 0.9,
        beta: float = 0.7,
        zeta: float = 0.1,
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
        self.zeta = zeta
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
        pi = self._PI(np.hstack(state))
        cur = self.rng.choice(len(self.action_set), p=pi.flatten())
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
        state = np.hstack(state)
        next_state = np.hstack(next_state)

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

    def _pi(self, state: Array) -> Array:
        return softmax(self.theta.T @ state)[None, :]


if __name__ == "__main__":
    from time import sleep
    from pathlib import Path
    import shutil

    import pandas as pd
    from tqdm.auto import trange

    from environment import Environment
    from plots import save_frames_as_gif
    from plots import metrics_plot, returns_plot

    def get_dir():
        return Path(config.BASE_PATH) / "00_central" / "{0:02d}".format(config.SEED)

    # Some helpful functions
    def plot_rewards(rewards, episodes) -> None:
        metrics_plot(
            rewards,
            "Average Rewards",
            "Train Rollouts (N={0}, seed={1:02d})".format(config.N_AGENTS, seed),
            save_directory_path=get_dir(),
            episodes=episodes,
        )

    def plot_mus(mus, episodes) -> None:
        metrics_plot(
            mus,
            "mu",
            "Train Mu (N={0}, seed={1:02d})".format(config.N_AGENTS, seed),
            rollouts=False,
            save_directory_path=get_dir(),
            episodes=episodes,
        )

    def plot_returns(rewards, episodes) -> None:
        returns_plot(
            rewards,
            episodes,
            "Train Returns (N={0}, seed={1:02d})".format(config.N_AGENTS, seed),
            save_directory_path=get_dir(),
        )

    def plot_eval(rewards) -> None:
        metrics_plot(
            rewards,
            "Average Rewards",
            "Evaluation Rollouts (N={0}, seed={1:02d})".format(config.N_AGENTS, seed),
            save_directory_path=get_dir(),
            episodes=[],
        )

    def plot_eval_best(rewards) -> None:
        metrics_plot(
            rewards,
            "Average Rewards",
            "Evaluation Rollouts (N={0}, seed={1:02d})".format(config.N_AGENTS, seed),
            save_directory_path=get_dir() / 'best',
        )

    def save(data, filename) -> None:
        csvname = "{0}.csv".format(filename)
        file_path = Path(get_dir()) / csvname
        pd.DataFrame(data=data.round(2)).to_csv(file_path.as_posix(), sep=",")

    seed = config.SEED
    env = Environment(
        n=config.N_AGENTS,
        scenario="networked_spread",
        seed=seed,
        central=ActorCriticCentral.fully_observable,
        randomize_reward_coefficients=config.RANDOMIZE_REWARD_COEFFICIENTS
    )
    agent = ActorCriticCentral(
        n_players=config.N_AGENTS,
        n_features=env.n_features,
        action_set=env.action_set,
        alpha=config.ALPHA,
        beta=config.BETA,
        zeta=config.ZETA,
        explore_episodes=config.EXPLORE_EPISODES,
        explore=config.EXPLORE,
        decay=False,
        seed=seed,
    )

    print("Fully observable: {0}".format(agent.fully_observable))
    print("Fully observable: {0}".format(ActorCriticCentral.fully_observable))
    print(agent.label)

    make_dirs(get_dir())
    episodes = []
    rewards = []
    mus = []
    best_rewards = -np.inf
    best_episode = 0
    for episode in trange(config.EPISODES, desc="episodes"):
        # execution loop
        obs = env.reset()
        actions = agent.act(obs)

        for _ in range(100):
            # step environment
            next_obs, next_rewards, _ = env.step(actions)

            # actor parameters.
            next_actions = agent.act(next_obs)

            agent.update(obs, actions, next_rewards, next_obs, next_actions)

            obs = next_obs
            actions = next_actions

            rewards.append(np.mean(next_rewards))
            episodes.append(episode)
            mus.append(agent.mu)

        if episode % 1000 == 0 or episode == config.EPISODES - 1:
            env.save_checkpoints(get_dir(), 'current')
            agent.save_checkpoints(get_dir(), 'current')

            eval_agent = ActorCriticCentral.load_checkpoint(get_dir(), 'current')
            eval_env = Environment.load_checkpoint(get_dir(), 'current')

            eval_rewards = 0
            first = True
            for _ in range(32):

                # Reseeds a copy of the original environment.
                eval_seed = 47 if first else None
                obs = eval_env.reset(seed=eval_seed)
                eval_agent.reset(seed=eval_seed)
                actions = eval_agent.act(obs)
                first = False
                for _ in trange(100, desc="timesteps"):
                    # step environment
                    next_obs, next_rewards, _ = env.step(actions)

                    # actor parameters.
                    next_actions = eval_agent.act(next_obs)

                    obs = next_obs
                    actions = next_actions
                    eval_rewards += np.mean(next_rewards)

            print('Evaluation: Current: {0:0.2f}\tBest: {1:0.2f}'.format(eval_rewards / 3200, best_rewards))
            if eval_rewards / 3200 > best_rewards:
                if not (get_dir() / 'best').exists():
                    (get_dir() / 'best').mkdir()

                if not (get_dir() / 'best' / str(episode)).exists():
                    (get_dir() / 'best' / str(episode)).mkdir()

                for chkpt_path in (get_dir() / 'current').glob('*.chkpt'):
                    if (get_dir() / 'best' / str(episode) / chkpt_path.name).exists():
                        (get_dir() / 'best' / str(episode) / chkpt_path.name).unlink()
                    shutil.move(chkpt_path.as_posix(), (get_dir() / 'best' / str(episode)).as_posix())

                if best_episode < episode:
                    shutil.rmtree((get_dir() / 'best' / str(best_episode)).as_posix())

                test_agent = ActorCriticCentral.load_checkpoint((get_dir() / 'best'), str(episode))
                np.testing.assert_array_almost_equal(test_agent.omega, agent.omega)
                np.testing.assert_array_almost_equal(test_agent.theta, agent.theta)

                best_rewards = eval_rewards / 3200
                best_episode = episode

    if (get_dir() / 'current').exists():
        shutil.rmtree((get_dir() / 'current').as_posix())
    # Train plots and saves data.
    plot_returns(rewards, episodes)
    plot_rewards(rewards, episodes)
    plot_mus(mus, episodes)

    pd.DataFrame(
        data=np.array(rewards).reshape((100, config.EPISODES)),
        columns=[*range(config.EPISODES)],
    ).to_csv(
        (Path(get_dir() / "train-seed{0:02d}.csv".format(config.SEED)).as_posix()),
        sep=",",
    )
        
    # This is a validation run.
    # Guarantees experiments are comparable.
    obs = env.reset(seed=config.SEED)
    prev_world = obs
    print("World state: {0}".format(obs))
    agent.explore = False
    agent.reset(seed=config.SEED)
    actions = agent.act(obs)
    frames = []
    rewards = []
    for _ in trange(100, desc="timesteps"):
        # env.render()  # for humans
        sleep(0.1)
        frames += env.render(mode="rgb_array")  # for saving

        # step environment
        next_obs, next_rewards, _ = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions

        rewards.append(np.mean(next_rewards))

    plot_eval(rewards)
    pd.DataFrame(data=np.array(rewards).reshape((100, 1)), columns=[1]).to_csv(
        (Path(get_dir()) / "test-seed{0:02d}.csv".format(config.SEED)).as_posix(),
        sep=",",
    )
    save_frames_as_gif(
        frames,
        dir_path=get_dir(),
        filename="simulation-seed{0:02d}.gif".format(seed),
    )

    shutil.copy("config.py", Path(get_dir()).as_posix())
    print('Evaluation rewards (LAST): {0:0.2f}'.format(np.sum(rewards)))

    # This is a validation run for best policy.
    # Guarantees experiments are comparable.
    obs = env.reset(seed=config.SEED)
    print("World state: {0}".format(obs))
    np.testing.assert_almost_equal(obs, prev_world)
    agent = ActorCriticCentral.load_checkpoint((get_dir() / 'best'), str(best_episode))
    agent.explore = False
    agent.reset(seed=config.SEED)
    actions = agent.act(obs)
    frames = []
    rewards = []
    for _ in trange(100, desc="timesteps"):
        # env.render()  # for humans
        sleep(0.1)
        frames += env.render(mode="rgb_array")  # for saving

        # step environment
        next_obs, next_rewards, _ = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions
        rewards.append(np.mean(next_rewards))

    plot_eval_best(rewards)
    pd.DataFrame(data=np.array(rewards).reshape((100, 1)), columns=[1]).to_csv(
        (Path(get_dir()) / "test-best-seed{0:02d}.csv".format(config.SEED)).as_posix(),
        sep=",",
    )
    save_frames_as_gif(
        frames,
        dir_path=get_dir(),
        filename="simulation-best-seed{0:02d}.gif".format(seed),
    )
    print('Evaluation rewards (BEST): {0:0.2f}'.format(np.sum(rewards)))
