#!/usr/bin/env python
"""Runs (a) train - evaluation loop (b) test (c) simulation."""
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union, Dict
from time import sleep

import numpy as np
import pandas as pd

import config
from central import ActorCriticCentral
from independent_learners import ActorCriticIndependent
from distributed_learners import ActorCriticDistributedV
from distributed_learners2 import ActorCriticDistributedQ
from consensus_learners import ActorCriticConsensus
from interfaces import AgentInterface

from environment import Environment
from common import Array, make_dirs, Action
from plots import train_plot, rollout_plot
from tqdm.auto import trange
from plots import save_frames_as_gif
from plots import metrics_plot
from log import log_traces


# Pipeline Types
Result = Tuple[int, Environment, AgentInterface, Sequence, Dict]
Results = List[Result]
RolloutCheckpoint = Tuple[int, float]
RolloutTest = Tuple[int, Array, Dict]
RolloutsTest = Tuple[RolloutTest]
RuR = Union[Results, RolloutsTest]
PATHS = {
    "ActorCriticCentral": "00_central",
    "ActorCriticDistributedV": "01_distributed_learners_v",
    "ActorCriticDistributedQ": "02_distributed_learners_q",
    "ActorCriticIndependent": "03_independent_learners",
    "ActorCriticConsensus": "04_consensus_learners",
}


def get_agent() -> AgentInterface:
    return eval(config.AGENT_TYPE)


def get_dir() -> Path:
    return Path(config.BASE_PATH) / PATHS[config.AGENT_TYPE] / "02"

def get_couplings(actions: Action) -> int:
    """Coupling: measures if all agents select the same action"""
    return int(len(set(actions)) == 1)

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


def rollout_w(args: Result) -> RolloutTest:
    """Thin wrapper for rollout.

    Parameters:
    -----------
    result: Result
        See train

    Returns:
    --------
    rollout: RolloutTest
        Comprising of num: int and res: Array

    See Also:
    ---------
    rollout: RolloutTest
    """
    return rollout_test(*args[:3])


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
        info: Dict
            Extra loop data, e.g, collisions and couplings.
    """
    # Defines the actor critic agent.
    Agent = get_agent()

    # Defines the environment
    env = Environment(
        n=config.N_AGENTS,
        scenario="networked_spread",
        seed=seed,
        central=Agent.fully_observable,
        randomize_reward_coefficients=config.RANDOMIZE_REWARD_COEFFICIENTS,
        communication=Agent.communication,
        cm_type=config.CONSENSUS_MATRIX_TYPE,
        cm_max_edges=config.CONSENSUS_MAX_EDGES
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

    # Make directories
    current_dir_path = get_dir() / "current"
    make_dirs(current_dir_path / str(seed))
    best_dir_path = get_dir() / "best"
    make_dirs(best_dir_path / str(seed))

    res = []
    info = defaultdict(list)
    best_rewards = -np.inf
    best_episode = 0
    chkpt_episode = 0
    for _ in trange(config.TRAINING_CYCLES, desc="pipeline_cycle"):

        train_checkpoint(num, env, agent, res, info)

        chkpt_episode += config.CHECKPOINT_INTERVAL
        # Save current checkpoints
        env.save_checkpoints(current_dir_path, str(seed))
        agent.save_checkpoints(current_dir_path, str(seed))

        # Call rollouts
        _, chkpt_rewards = rollout_checkpoint(num, current_dir_path, seed)

        if chkpt_rewards > best_rewards:
            chkpt_dir_path = best_dir_path / str(seed) / str(chkpt_episode)
            make_dirs(chkpt_dir_path)

            for chkpt_path in (current_dir_path / str(seed)).glob("*.chkpt"):
                if (chkpt_dir_path / chkpt_path.name).exists():
                    (chkpt_dir_path / chkpt_path.name).unlink()
                shutil.move(chkpt_path.as_posix(), chkpt_dir_path.as_posix())

            if best_episode <= chkpt_episode:
                chkpt_best_dir_path = best_dir_path / str(seed) / str(best_episode)
                if chkpt_best_dir_path.exists():
                    shutil.rmtree(chkpt_best_dir_path.as_posix())

            best_rewards = chkpt_rewards
            best_episode = chkpt_episode
            chkpt_best_dir_path = best_dir_path / str(seed) / str(best_episode)

    # Save best checkpoint
    # Result is a column array
    res = np.array(res)[:, None]
    chkpt_agent = Agent.load_checkpoint(chkpt_best_dir_path, '')

    return (num, env, chkpt_agent, res, info)

def train_checkpoint(
    num: int, env: Environment, agent: AgentInterface, res: List, info: Dict
) -> Result:
    """Trains agent for CHECKPOINT_INTERVAL steps"""
    # Starts the training
    for _ in trange(config.CHECKPOINT_INTERVAL, desc="train_episodes"):

        # execution loop
        obs = env.reset()
        agent.reset()
        actions = agent.act(obs)

        rewards = []
        for _ in trange(100, desc="train_timesteps"):
            info["couplings"].append(get_couplings(actions))

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
            info["collisions"].append(env.n_collisions())
        res.append(np.sum(rewards))
    return num, env, agent, res, info


def rollout_checkpoint(
    num: int, current_dir_path: Path, seed: int
) -> RolloutCheckpoint:
    """Runs an evaluation run with the same environment.

    Parameters:
    -----------
    um: int,
    current_dir_path: Path
    seed: int

    Returns:
    --------
    rollout: RolloutCheckpoint
        The rollout type -- num: int and rewards: Array
    """
    # Defines the actor critic agent.
    Agent = get_agent()

    # Loads from disk the environment and agent
    # Guarantees a copy is being generated.
    agent_chkpt = Agent.load_checkpoint(current_dir_path, str(seed))
    env_chkpt = Environment.load_checkpoint(current_dir_path, str(seed))
    agent_chkpt.explore = False

    res = []
    for i in trange(config.CHECKPOINT_EVALUATIONS, desc="checkpoint_evaluations"):
        eval_seed = 2022 if (i == 0) else None
        obs = env_chkpt.reset(seed=eval_seed)
        agent_chkpt.reset(seed=eval_seed)
        actions = agent_chkpt.act(obs)
        for _ in trange(100, desc="checkpoint_timesteps"):

            # step environment
            next_obs, next_rewards, _ = env_chkpt.step(actions)

            next_actions = agent_chkpt.act(next_obs)

            obs = next_obs
            actions = next_actions
            res.append(np.mean(next_rewards))

    # Result is a column array
    res = np.mean(res)
    return (num, res)


def rollout_test(num: int, env: Environment, agent: AgentInterface) -> RolloutTest:
    """Runs a single rollout for testing.

    Parameters:
    -----------
    num: int
    env: Environment
    agent: Agent

    Returns:
    --------
    rollout: RolloutTest
        The rollout type -- num: int and rewards: Array
    """
    # Makes a copy to not change internal states.
    rollout_env = deepcopy(env)
    rollout_agent = deepcopy(agent)

    obs = rollout_env.reset(seed=env.scenario.seed + config.ROLLOUT_SEED_INC)
    rollout_agent.reset()
    actions = rollout_agent.act(obs)
    res = []
    info = defaultdict(list)
    for _ in trange(100, desc="timesteps"):
        info["couplings"].append(get_couplings(actions))

        # step environment
        next_obs, next_rewards, _ = rollout_env.step(actions)

        next_actions = rollout_agent.act(next_obs)

        obs = next_obs
        actions = next_actions
        res.append(np.mean(next_rewards))

        info["collisions"].append(rollout_env.n_collisions())
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


def top_k(tuples_list: RuR, k: int = 5, skip_obs: int = 0) -> RuR:
    """Returns the top k experiments

    Parameters:
    -----------
    tuples_list: union[results, rollouts]
        Either results or rollouts list of tuples.
    k: int = 5
        Top K results
    skip_obs: int = 0
        Skips the first `skip_obs` observations.
        For rollouts will skip timesteps for
        results will skip episodes.

    Returns:
    --------
    tuples_list: union[results, rollouts]
        Returns the k-best experiments according to returns
    """

    def fn(x):
        return np.mean(x[-2][skip_obs:])

    return sorted(tuples_list, key=fn, reverse=True)[:k]


def simulate(
    num: int,
    env: Environment,
    agent: AgentInterface,
    save_directory_path: Path = None,
    render: bool = False,
) -> float:
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
    # Makes a copy to not change internal states.
    sim_env = deepcopy(env)
    sim_agent = deepcopy(agent)

    obs = sim_env.reset(seed=env.scenario.seed + config.ROLLOUT_SEED_INC)
    sim_agent.reset()
    actions = sim_agent.act(obs)
    frames = []
    rewards = []
    for _ in trange(100, desc="timesteps"):
        if render:
            sleep(0.1)
            frames += sim_env.render(mode="rgb_array")  # for saving

        next_obs, next_rewards, _ = sim_env.step(actions)

        next_actions = sim_agent.act(next_obs)

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

    return np.mean(rewards)

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
        log_traces(seed, x0, a0, r1, x1, get_dir())


if __name__ == "__main__":
    from operator import itemgetter
    from multiprocessing.pool import Pool
    from pathlib import Path
    import shutil

    # Make preliminary directories
    make_dirs(get_dir() / "current")
    make_dirs(get_dir() / "best")

    # Make directory here.
    with Pool(config.N_WORKERS) as pool:
        results = pool.map(train_w, enumerate(config.PIPELINE_SEEDS))

    train_plot(get_results(results), n=config.N_AGENTS, save_directory_path=get_dir())

    pd.DataFrame(data=get_results(results), columns=config.PIPELINE_SEEDS).to_csv(
        (get_dir() / "pipeline-train.csv").as_posix(), sep=","
    )
    pd.DataFrame(
        data=get_results(results), columns=config.PIPELINE_SEEDS
    ).describe().to_csv((get_dir() / "pipeline-train-summary.csv").as_posix(), sep=",")

    # Counts the number of times the
    # same action was selected by both agents.
    couplings = np.vstack([res[-1]["couplings"] for res in results]).T
    pd.DataFrame(data=couplings, columns=config.PIPELINE_SEEDS).to_csv(
        (get_dir() / "pipeline-train-couplings.csv").as_posix(), sep=","
    )

    # Counts the number of times the
    # agents collided on a single timestep
    collisions = np.vstack([res[-1]["collisions"] for res in results]).T
    pd.DataFrame(data=collisions, columns=config.PIPELINE_SEEDS).to_csv(
        (get_dir() / "pipeline-train-collisions.csv").as_posix(), sep=","
    )

    results_k = top_k(results, k=3)
    train_plot(get_results(results_k))
    pd.DataFrame(
        data=get_results(results_k),
        columns=[config.PIPELINE_SEEDS[rok[0]] for rok in results_k],
    ).to_csv((get_dir() / "pipeline-results-top03.csv").as_posix(), sep=",")

    # Rollouts
    rollouts = [*map(rollout_w, results)]
    # Rollouts plot
    rollout_plot(
        get_results(rollouts), n=config.N_AGENTS, save_directory_path=get_dir()
    )
    pd.DataFrame(data=get_results(rollouts), columns=config.PIPELINE_SEEDS).to_csv(
        (get_dir() / "pipeline-rollouts.csv").as_posix(), sep=","
    )
    pd.DataFrame(
        data=get_results(rollouts), columns=config.PIPELINE_SEEDS
    ).describe().to_csv(
        (get_dir() / "pipeline-rollouts-summary.csv").as_posix(), sep=","
    )
    # Counts the number of times the
    # same action was selected by both agents.
    couplings = np.vstack([res[-1]["couplings"] for res in rollouts]).T
    pd.DataFrame(data=couplings, columns=config.PIPELINE_SEEDS).to_csv(
        (get_dir() / "pipeline-rollouts-couplings.csv").as_posix(), sep=","
    )

    # Counts the number of times the
    # agents collided on a single timestep
    collisions = np.vstack([res[-1]["collisions"] for res in rollouts]).T
    pd.DataFrame(data=collisions, columns=config.PIPELINE_SEEDS).to_csv(
        (get_dir() / "pipeline-rollouts-collisions.csv").as_posix(), sep=","
    )
    # Rollouts plot -- K Best runs.
    rollouts_k = top_k(rollouts, k=3)
    rollout_plot(
        get_results(rollouts_k), n=config.N_AGENTS, save_directory_path=get_dir()
    )
    pd.DataFrame(
        data=get_results(rollouts_k),
        columns=[config.PIPELINE_SEEDS[rok[0]] for rok in rollouts_k],
    ).to_csv((get_dir() / "pipeline-rollouts-top03.csv").as_posix(), sep=",")

    shutil.copy("config.py", get_dir().as_posix())
    def fn(x):
        return rollouts_k[0][0] == x[0]
    result_from_best_rollout = [*filter(fn, results)][0]
    max_averaged_rewards = np.mean(rollouts_k[0][1])
    sim_averaged_rewards = simulate(*result_from_best_rollout[:3], save_directory_path=get_dir(), render=False)
    np.testing.assert_almost_equal(max_averaged_rewards, sim_averaged_rewards)
    print('max_average_reward:{0:0.4f}\tsimulation_reward:{1:0.4f}'.format(max_averaged_rewards, sim_averaged_rewards))

