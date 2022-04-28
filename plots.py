"""Ploting helper moodule. """
from typing import List
from common import Array

import numpy as np
import matplotlib.pyplot as plt

FIGURE_X = 6.0
FIGURE_Y = 4.0


def rewards_plot(rewards: List[float], episodes: List[int]) -> None:
    """Plots the reward for first, mid, last episode"""

    rewards = np.array(rewards)
    episodes = np.array(episodes)

    Y = []
    first, mid, last = (
        int(np.min(episodes)),
        int(np.mean(episodes)),
        int(np.max(episodes)),
    )
    for idx in (first, mid, last):
        Y.append(rewards[episodes == idx])
    Y = np.vstack(Y).T

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    labels = tuple(["episode %s" % idx for idx in (first, mid, last)])
    for idx in range(3):
        plt.plot(X, Y[:, idx], label=labels[idx])
    plt.xlabel("Time")
    plt.ylabel("Average Reward")
    plt.legend(loc=4)
    plt.show()


def returns_plot(rewards: List[float], episodes: List[int]) -> None:
    """Plots the returns"""

    rewards = np.array(rewards)
    episodes = np.array(episodes)

    Y = []
    for idx in sorted(set(episodes)):
        Y.append(np.sum(rewards[episodes == idx]))
    Y = np.stack(Y)

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.plot(X, Y)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward Return")
    plt.legend(loc=4)
    plt.show()


def train_plot(results: Array, n: int = 1) -> None:
    """Plots the results from a training run.

    Parameters:
    -----------
    results: Array
        A N x M array with train returns, where:
            N: The number of episodes.
            M: The number of independent runs.
    n: int = 1
        The number of agents
    """
    _, M = results.shape
    Y = np.mean(results, axis=1)
    Y_std = np.std(results, axis=1)

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)

    fig, axis = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    axis.plot(X, Y, c="C0")
    axis.fill_between(X, Y - Y_std, Y + Y_std, facecolor="C0", alpha=0.5)
    plt.suptitle("Train (N=%s, M=%s)" % (n, M))
    plt.xlabel("Episode")
    plt.ylabel("Average Reward Return")
    plt.show()


def rollout_plot(results: Array, n: int = 1) -> None:
    """Plots the results from a rollout run.

    Parameters:
    -----------
    results: Array
        A N x M array with train returns, where:
            N: The number of timesteps.
            M: The number of independent runs.

    n: int = 1
        The number of agents
    """
    _, M = results.shape
    Y = np.mean(results, axis=1)
    Y_std = np.std(results, axis=1)

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)

    fig, axis = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    axis.plot(X, Y, c="C0")
    axis.fill_between(X, Y - Y_std, Y + Y_std, facecolor="C0", alpha=0.5)
    plt.suptitle("Rollouts (N=%s, M=%s)" % (n, M))
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.show()
