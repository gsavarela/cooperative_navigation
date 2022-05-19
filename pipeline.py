#!/usr/bin/env python
"""Runs (a) train (b) evaluation (c) simulation.

TODO:
-----
* introduce interfaces in the project as types.

"""
from pathlib import Path
from typing import List, Tuple, Union
from time import sleep
import numpy as np
import pandas as pd


import config
from central import ActorCriticCentral
from independent_learners import ActorCriticIL
from distributed_learners import ActorCriticDistributed
from interfaces import AgentInterface

from environment import Environment
from common import Observation, Action, Rewards, Array
from plots import train_plot, rollout_plot
from tqdm.auto import trange
from plots import save_frames_as_gif
from plots import metrics_plot


# Pipeline Types
Trace = Tuple[Observation, Action, Rewards, Observation, Action]
Traces = List[Trace]
Result = Tuple[int, Environment, AgentInterface, Traces, Array]
Results = List[Result]
Rollout = Tuple[int, Array]
Rollouts = Tuple[Rollout]
RuR = Union[Results, Rollouts]
PATHS = {
    "ActorCriticCentral": "00_central",
    "ActorCriticDistributed": "01_distributed_learners",
    "ActorCriticIL": "02_independent_learners",
}


def get_agent() -> AgentInterface:
    return eval(config.AGENT_TYPE)


def get_dir() -> str:
    return config.BASE_PATH + "/" + PATHS[config.AGENT_TYPE]


def train_w(args: Tuple[int]) -> Result:
    """Thin wrapper for train.

    Parameters:
    -----------
    num: int
        The experiment identifier.
    seed: int
        Regulates the random number generator.

    Returns:
    --------
    result: Result
        Definition:
        num: int
            The experiment number
        env: Environment
            The environment encapsulating a scenario.
        agent: Agent
            The reinforcement learning agent.
        traces: Traces
            The tuple (s, a, r, s', a')
        results: Array
            The returns from the episodes.

    See Also:
    ---------
    train: Result
    """
    return train(*args)


def rollout_w(args: Result) -> Rollout:
    """Thin wrapper for rollout.

    Parameters:
    -----------
    result: Result
        See train

    Returns:
    --------
    rollout: Rollout
        Comprising of num: int and res: Array

    See Also:
    ---------
    rollout: Rollout
    """
    return rollout(*args[:3])


def train(num: int, seed: int) -> Result:
    """Estimate the policy using a reinforcement learning agent.

    Parameters:
    -----------
    num: int
        The experiment identifier.

    seed: int
        Regulates the random number generator.

    Returns:
    --------
    result: Result
        Defined as:
        num: int
            The experiment number
        env: Environment
            The environment encapsulating a scenario.
        agent: Agent
            The reinforcement learning agent.
        traces: Traces
            The tuple (s, a, r, s', a')
        results: Array
            The returns from the episodes.
    """
    # Defines the environment
    env = Environment(
        n=config.N_AGENTS,
        scenario="networked_spread",
        seed=seed,
        central=True,
    )

    # Defines the actor critic agent.
    Agent = get_agent()
    agent = Agent(
        n_players=env.n,
        n_features=env.n_features,
        action_set=env.action_set,
        alpha=config.ALPHA,
        beta=config.BETA,
        explore_episodes=config.EXPLORE_EPISODES,
        explore=config.EXPLORE,
        decay=False,
        seed=seed,
    )
    first = True
    res = []
    traces = []
    # Starts the training
    for _ in trange(config.EPISODES, desc="episodes"):
        # execution loop
        obs = env.reset()

        if not first:
            agent.reset()
        actions = agent.act(obs)

        first = False
        rewards = []
        for _ in trange(100, desc="timesteps"):
            # step environment
            next_obs, next_rewards = env.step(actions)

            next_actions = agent.act(next_obs)

            tr = (obs, actions, next_rewards, next_obs, next_actions)

            agent.update(*tr)

            obs = next_obs
            actions = next_actions
            rewards.append(np.mean(next_rewards))
            traces.append(tr)
        res.append(np.sum(rewards))
    # Result is a column array
    res = np.array(res)[:, None]

    return (num, env, agent, traces, res)


def rollout(num: int, env: Environment, agent: AgentInterface) -> Rollout:
    """Runs an evaluation run with the same environment.

    Parameters:
    -----------
    num: int
    env: Environment
    agent: Agent

    Returns:
    --------
    rollout: Rollout
        The rollout type -- num: int and rewards: Array
    """
    obs = env.reset()
    agent.reset(seed=env.scenario.seed)
    agent.explore = False
    actions = agent.act(obs)
    res = []
    for _ in trange(100, desc="timesteps"):

        # step environment
        next_obs, next_rewards = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions
        res.append(np.mean(next_rewards))

    # Result is a column array
    res = np.array(res)[:, None]
    return (num, res)


def get_results(tuples_list: RuR) -> Array:
    """Strips either the returns or rewards.

    Parameters:
    -----------
    tuples_list: Union[results, rollouts]
        Either results or rollouts list of tuples.

    Returns:
    --------
    Returns or Rewards: Array
        The time series of Returns (Results) or Rewards (Rollouts)
    """
    return np.hstack([*map(itemgetter(-1), tuples_list)])


def top_k(tuples_list: RuR, k: int = 5) -> RuR:
    """Returns the top k experiments

    Parameters:
    -----------
    tuples_list: union[results, rollouts]
        Either results or rollouts list of tuples.

    Returns:
    --------
    tuples_list: union[results, rollouts]
        Returns the k-best experiments according to returns
    """

    def fn(x):
        return np.mean(x[-1][50:])

    return sorted(tuples_list, key=fn, reverse=True)[:k]


def simulate(
    num: int, env: Environment, agent: AgentInterface, save_directory_path: Path = None
) -> None:
    """Renders the experiment for 100 timesteps.

    Parameters:
    -----------
    num: int
        The experiment identifier
    env: Environment
        The environment used to run.
    agent: AgentInterface
        The reinforcement learning agent (controls one or more players).
    """
    obs = env.reset()
    agent.reset()
    agent.explore = False
    actions = agent.act(obs)
    frames = []
    rewards = []
    for _ in trange(100, desc="timesteps"):
        sleep(0.1)
        frames += env.render(mode="rgb_array")  # for saving

        next_obs, next_rewards = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions
        rewards.append(np.mean(next_rewards))

    if save_directory_path is not None:
        metrics_plot(
            rewards,
            "Average Rewards",
            "Evaluation Rollout (N={0}, num={1:02d})".format(config.N_AGENTS, num),
            save_directory_path=save_directory_path,
            episodes=[],
        )
        pd.DataFrame(data=np.array(rewards).reshape((100, 1)), columns=[1]).to_csv(
            ("{0}/evaluation_rollout-num{1:02d}.csv".format(save_directory_path, num)), sep=","
        )
        save_frames_as_gif(
            frames,
            dir_path=save_directory_path,
            filename="simulation-pipeline-best.gif",
        )


if __name__ == "__main__":
    from operator import itemgetter
    from multiprocessing.pool import Pool
    from pathlib import Path
    import shutil

    target_dir = Path(get_dir()) / "02"
    with Pool(config.N_WORKERS) as pool:
        results = pool.map(train_w, enumerate(config.PIPELINE_SEEDS))
    train_plot(get_results(results), n=config.N_AGENTS, save_directory_path=target_dir)

    results_k = top_k(results, k=3)
    train_plot(get_results(results_k))

    rollouts = [*map(rollout_w, results)]
    rollout_plot(get_results(rollouts), n=config.N_AGENTS, save_directory_path=target_dir)
    pd.DataFrame(
        data=get_results(rollouts), columns=config.PIPELINE_SEEDS
    ).describe().to_csv((target_dir / "pipeline.csv").as_posix(), sep=",")

    rollouts_k = [
        roll for roll in rollouts if roll[0] in set([*map(itemgetter(0), results_k)])
    ]
    rollout_plot(get_results(rollouts_k), n=config.N_AGENTS, save_directory_path=target_dir)

    simulate(*results_k[0][:3], save_directory_path=target_dir)

    shutil.copy("config.py", target_dir.as_posix())
