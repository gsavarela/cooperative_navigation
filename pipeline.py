#!/usr/bin/env python
"""Runs (a) train (b) evaluation (c) simulation."""
from pathlib import Path
from typing import List, Tuple, Union
from time import sleep
import numpy as np
import pandas as pd
from collections import defaultdict


import config
from central import ActorCriticCentral
from independent_learners import ActorCriticIndependent
from distributed_learners2 import ActorCriticDistributed
from consensus_learners import ActorCriticConsensus
from interfaces import AgentInterface

from environment import Environment
from common import Observation, Action, Rewards, Array
from plots import train_plot, rollout_plot
from tqdm.auto import trange
from plots import save_frames_as_gif
from plots import metrics_plot
from log import log_traces


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
    "ActorCriticDistributed": "01_distributed_learners2",
    "ActorCriticIndependent": "02_independent_learners",
    "ActorCriticConsensus": "03_consensus_learners",
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
        results: Array
            The returns from the episodes.
    """
    # Defines the actor critic agent.
    Agent = get_agent()

    # Defines the environment
    env = Environment(
        n=config.N_AGENTS,
        scenario="networked_spread",
        seed=seed,
        central=Agent.fully_observable,
        communication=Agent.communication,
        cm_type=config.CONSENSUS_MATRIX_TYPE
    )

    # Instanciates the actor critic agent.
    agent = Agent(
        n_players=env.n,
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
    first = True
    res = []
    info = defaultdict(list)
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
            info['couplings'].append(int(len(set(actions))  == 1))

            # step environment
            next_obs, next_rewards, cwm = env.step(actions)

            next_actions = agent.act(next_obs)

            tr = (obs, actions, next_rewards, next_obs, next_actions)
            if agent.communication:
                tr += (cwm,)

            agent.update(*tr)

            obs = next_obs
            actions = next_actions
            rewards.append(np.mean(next_rewards))
            info['collisions'].append(env.n_collisions())
        res.append(np.sum(rewards))
    # Result is a column array
    res = np.array(res)[:, None]

    return (num, env, agent, res, info)


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
    info = defaultdict(list)
    for _ in trange(100, desc="timesteps"):
        info['couplings'].append(len(actions) - len(set(actions)))

        # step environment
        next_obs, next_rewards, _ = env.step(actions)

        next_actions = agent.act(next_obs)

        obs = next_obs
        actions = next_actions
        res.append(np.mean(next_rewards))

        info['collisions'].append(env.n_collisions())
    # Result is a column array
    res = np.array(res)[:, None]
    return (num, res, info)


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
    return np.hstack([*map(itemgetter(-2), tuples_list)])


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
        return np.mean(x[-2][50:])

    return sorted(tuples_list, key=fn, reverse=True)[:k]


def simulate(
        num: int, env: Environment, agent: AgentInterface, save_directory_path: Path = None, render: bool=False
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

    render: Bool, False
    """
    obs = env.reset()
    agent.reset()
    agent.explore = False
    actions = agent.act(obs)
    frames = []
    rewards = []
    for _ in trange(100, desc="timesteps"):
        if render:
            sleep(0.1)
            frames += env.render(mode="rgb_array")  # for saving

        next_obs, next_rewards, _ = env.step(actions)

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
            ("{0}/evaluation_rollout-num{1:02d}.csv".format(save_directory_path, num)),
            sep=",",
        )
        if render:
            save_frames_as_gif(
                frames,
                dir_path=save_directory_path,
                filename="simulation-pipeline-best.gif",
            )


def save_traces(results: Results, path: Path):
    """Convert every result in the list to a dataframe

    Parameters
    ----------
    results: Results
        The list of tuples (num, env, agent, traces, res)
    path: Path
        The saving path

    """
    for num, _, agent, traces, _, _ in results:
        seed = config.PIPELINE_SEEDS[num]
        x0, a0, r1, x1, _ = zip(*traces)
        if agent.fully_observable:
            x0 = [*map(np.hstack, x0)]
            x1 = [*map(np.hstack, x1)]
            r1 = [*map(np.mean, r1)]
        log_traces(seed, x0, a0, r1, x1, target_dir)

if __name__ == "__main__":
    from operator import itemgetter
    from multiprocessing.pool import Pool
    from pathlib import Path
    import shutil

    # Make directory here.
    target_dir = Path(get_dir()) / "02"
    with Pool(config.N_WORKERS) as pool:
        results = pool.map(train_w, enumerate(config.PIPELINE_SEEDS))

    train_plot(get_results(results), n=config.N_AGENTS, save_directory_path=target_dir)

    pd.DataFrame(data=get_results(results), columns=config.PIPELINE_SEEDS).to_csv(
        (target_dir / "pipeline-train.csv").as_posix(), sep=","
    )
    pd.DataFrame(
        data=get_results(results), columns=config.PIPELINE_SEEDS
    ).describe().to_csv((target_dir / "pipeline-train-summary.csv").as_posix(), sep=",")

    # Counts the number of times the
    # same action was selected by both agents.
    couplings = np.vstack([res[-1]['couplings'] for res in results]).T
    pd.DataFrame(data=couplings, columns=config.PIPELINE_SEEDS).to_csv(
        (target_dir / "pipeline-train-couplings.csv").as_posix(), sep=","
    )

    # Counts the number of times the
    # agents collided on a single timestep
    collisions = np.vstack([res[-1]['collisions'] for res in results]).T
    pd.DataFrame(data=collisions, columns=config.PIPELINE_SEEDS).to_csv(
        (target_dir / "pipeline-train-collisions.csv").as_posix(), sep=","
    )

    results_k = top_k(results, k=3)
    train_plot(get_results(results_k))
    pd.DataFrame(
        data=get_results(results_k),
        columns=[config.PIPELINE_SEEDS[rok[0]] for rok in results_k],
    ).to_csv((target_dir / "pipeline-results-best.csv").as_posix(), sep=",")


    # Rollouts
    rollouts = [*map(rollout_w, results)]
    # Rollouts plot
    rollout_plot(
        get_results(rollouts), n=config.N_AGENTS, save_directory_path=target_dir
    )
    pd.DataFrame(data=get_results(rollouts), columns=config.PIPELINE_SEEDS).to_csv(
        (target_dir / "pipeline-rollouts.csv").as_posix(), sep=","
    )
    pd.DataFrame(
        data=get_results(rollouts), columns=config.PIPELINE_SEEDS
    ).describe().to_csv(
        (target_dir / "pipeline-rollouts-summary.csv").as_posix(), sep=","
    )
    # Counts the number of times the
    # same action was selected by both agents.
    couplings = np.vstack([res[-1]['couplings'] for res in rollouts]).T
    pd.DataFrame(data=couplings, columns=config.PIPELINE_SEEDS).to_csv(
        (target_dir / "pipeline-rollouts-couplings.csv").as_posix(), sep=","
    )

    # Counts the number of times the
    # agents collided on a single timestep
    collisions = np.vstack([res[-1]['collisions'] for res in rollouts]).T
    pd.DataFrame(data=collisions, columns=config.PIPELINE_SEEDS).to_csv(
        (target_dir / "pipeline-rollouts-collisions.csv").as_posix(), sep=","
    )
    # Rollouts plot -- K Best runs.
    rollouts_k = [
        roll for roll in rollouts if roll[0] in set([*map(itemgetter(0), results_k)])
    ]
    rollout_plot(
        get_results(rollouts_k), n=config.N_AGENTS, save_directory_path=target_dir
    )
    pd.DataFrame(
        data=get_results(rollouts_k),
        columns=[config.PIPELINE_SEEDS[rok[0]] for rok in rollouts_k],
    ).to_csv((target_dir / "pipeline-rollouts-best.csv").as_posix(), sep=",")

    simulate(*results_k[0][:3], save_directory_path=target_dir, render=True)

    shutil.copy("config.py", target_dir.as_posix())
