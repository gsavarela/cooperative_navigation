"""One-step ActorCritic with Advantage For Continuing tasks.
    * Consensus.
    * Continuing tasks.
    * V function approximation.
    * Linear function approximation.
    * Fully observable setting.
    * Each agent learns it's own rewards.
    * Individual rewards.

References:
-----------
..[1] Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Basar
  2018. "Fully Decentralized Multi-player Re-inforcement Learning with
  Networked players". Proceedings of the 35th International Conference
  on Machine Learning, PMLR 80 (2018), 5872–5881.

..[2] Sutton and Barto 2018. "Introduction to Reinforcement
  Learning 2nd Edition" (pg 333).

..[3] Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000).
  "Policy gradient methods for reinforcement learning with function
  approximation". In Advances in neural information processing
  systems (NIPS), Denver, CO. Cambridge: MIT Press.
"""
import numpy as np

import config
from common import Array, Observation, Action, ActionSet, Rewards
from common import softmax, make_dirs
from interfaces import AgentInterface, ActorCriticInterface, SerializableInterface


def clip(x):
    return np.clip(x, -1, 1)


class ActorCriticConsensus(AgentInterface, ActorCriticInterface, SerializableInterface):
    """Consensus actor critic with Linear function approximation

    Consensus critic and independent actors.
    * The critic estimates a Q(s,a) function.

    Attributes
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

    Methods
    -------
    act(state): Action
        Select an action based on state

    reset(seed): None
        Resets seed, updates number of steps.

    update(state, actions, next_reward, next_state): None
        Learns from policy improvement and policy evalution.

    """

    fully_observable = True
    communication = True

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
        self.n_actions = int(np.round(np.power(len(action_set), 1 / n_players), 0))
        self.n_features = n_features

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        self.omega = np.zeros((self.n_players, self.n_features, len(self.action_set)))
        self.theta = np.zeros((self.n_players, self.n_features, self.n_actions))
        self.zeta = zeta
        self.mu = np.zeros(self.n_players)

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
            ret.append(self.rng.choice(5, p=pi.flatten()))
        return tuple(ret)

    def update(
        self,
        state: Observation,
        actions: Action,
        next_rewards: Rewards,
        next_state: Observation,
        next_actions: Action,
        cwm: Array,
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
        cwm: Array
            The consensus weights matrix for this episode.
        """

        # Aggregate states and instante next omega
        _s0 = np.hstack(state)
        _s1 = np.hstack(next_state)
        _a0 = self.action_set.index(actions)
        _a1 = self.action_set.index(next_actions)
        _o1 = self.omega.copy()  # The omega for the next iteration

        for _n in range(self.n_players):
            # Gets individual action
            _a0n = actions[_n]

            # Computes TD(0) Error.
            delta = clip(
                (next_rewards[_n] - self.mu[_n])
                + (_s1 @ self.omega[_n, :, _a1] - _s0 @ self.omega[_n, :, _a0]),
            )
            # Critic before consensus
            _o1[_n, :, _a0] = clip(_o1[_n, :, _a0] + self.alpha * delta * _s0)

            # Actor
            self.theta[_n] += (
                self.beta * self._A(_s0, _a0, _n) * self._psi(_s0, _a0n, _n)
            )

            self.mu[_n] = (1 - self.zeta) * self.mu[_n] + self.zeta * next_rewards[_n]
        self.omega = np.einsum("ij, jmk -> imk", cwm, _o1)

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

    def _A(self, state: Array, action: int, n: int) -> Array:
        _qx = state @ self.omega[n, :, action]

        actions = list(self.action_set[action])
        _pi = self._PI(state, n)
        _vn = 0
        for _an in range(self.n_actions):
            _actions = actions[:]
            _actions[n] = _an
            _a = self.action_set.index(tuple(_actions))
            _vn += state @ self.omega[n, :, _a] * _pi[0, _an]
        return _qx - _vn


if __name__ == "__main__":
    from time import sleep
    from pathlib import Path
    import shutil

    import pandas as pd
    from tqdm.auto import trange

    from plots import save_frames_as_gif
    from plots import metrics_plot, returns_plot
    from environment import Environment

    def get_dir():
        return (
            Path(config.BASE_PATH)
            / "03_consensus_learners"
            / "{0:02d}".format(config.SEED)
        )

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
        central=ActorCriticConsensus.fully_observable,
        communication=ActorCriticConsensus.communication,
        cm_type=config.CONSENSUS_MATRIX_TYPE,
        cm_max_edges=config.CONSENSUS_MAX_EDGES
    )
    agent = ActorCriticConsensus(
        n_players=config.N_AGENTS,
        n_features=env.n_features,
        action_set=env.action_set,
        alpha=config.ALPHA,
        beta=config.BETA,
        zeta=config.ZETA,
        explore_episodes=config.EXPLORE_EPISODES,
        explore=config.EXPLORE,
        decay=False,
        seed=config.SEED,
    )

    print("Fully observable: {0}".format(agent.fully_observable))
    print("Fully observable: {0}".format(ActorCriticConsensus.fully_observable))
    print("Communication: {0}".format(ActorCriticConsensus.communication))
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

        for _ in trange(100, desc="timesteps"):
            # step environment
            next_obs, next_rewards, cwm = env.step(actions)

            # actor parameters.
            next_actions = agent.act(next_obs)

            agent.update(obs, actions, next_rewards, next_obs, next_actions, cwm)

            obs = next_obs
            actions = next_actions

            rewards.append(np.mean(next_rewards))
            episodes.append(episode)
            mus.append(np.mean(agent.mu))

        if episode % 1000 == 0 or episode == config.EPISODES - 1:
            env.save_checkpoints(get_dir(), 'current')
            agent.save_checkpoints(get_dir(), 'current')

            eval_agent = ActorCriticConsensus.load_checkpoint(get_dir(), 'current')
            eval_env = Environment.load_checkpoint(get_dir(), 'current')

            eval_rewards = 0
            first = True
            for _ in trange(32, desc="evaluation"):

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

                test_agent = ActorCriticConsensus.load_checkpoint((get_dir() / 'best'), str(episode))
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
    agent.reset(seed=config.SEED)
    agent.explore = False
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
    agent = ActorCriticConsensus.load_checkpoint((get_dir() / 'best'), str(best_episode))
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



