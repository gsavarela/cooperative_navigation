"""Ploting helper moodule. """
import string
from pathlib import Path
import re
from typing import List
from common import Array

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import statsmodels.api as sm  # Seaborn to make fast computations.
import gym

FIGURE_X = 6.0
FIGURE_Y = 4.0
MEAN_CURVE_COLOR = (0.89, 0.282, 0.192)
SMOOTHING_CURVE_COLOR = (0.33, 0.33, 0.33)
SEED_PATTERN = r"seed=(.*?)\)"
M_PATTERN = r"M=(.*?)\,"

def save_frames_as_gif(frames: List[Array], dir_path: Path = None, filename: str = 'simulation.gif') -> None:
    """Saves a list of frames into gif format.

    * Ensure you have imagemagick installed with
    > sudo apt-get install imagemagick

    * Open file in CLI with:
    xgd-open <filename>

    Parameters:
    -----------
    frames: List[Array],
        A list of images.
    dir_path: str, default None
        The directory to save animations, reverts to './data'
    filename: str, default 'simulation.gif'
        The GIF file name.
    """

    if dir_path is None:
        dir_path = Path('data/')

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save((dir_path / filename).as_posix(), writer='imagemagick', fps=60)


def _savefig(suptitle: str, save_directory_path: Path = None) -> None:
    """Saves a figure, named after suptitle, if save_directory_path is provided

    Parameters:
    ----------
    suptitle: str
        The title.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.

    Returns:
    -------
    filename: str
        Space between words are filled with underscore (_).
    """
    if save_directory_path is not None:
        # iteractively check for path and builds where it doesnt exist.
        prev_path = None
        for sub_dir in save_directory_path.parts:
            if prev_path is None:
                prev_path = Path(sub_dir)
            else:
                prev_path = prev_path / sub_dir
            prev_path.mkdir(exist_ok=True)
        # uses directory
        file_path = save_directory_path / "{0}.png".format(_to_filename(suptitle))
        plt.savefig(file_path.as_posix())


def _snakefy(title_case: str) -> str:
    """Converts `Title Case` into `snake_case`

    Parameters:
    ----------
    title_case: str
        Uppercase for new words and spaces to split then up.

    Returns:
    -------
    filename: str
        Space between words are filled with underscore (_).
    """
    fmt = title_case.translate(str.maketrans("", "", string.punctuation))
    return "_".join(fmt.lower().split())


def _to_filename(suptitle: str) -> str:
    """Formats a plot title to filenme"""

    # Tries to search for a seed pattern and than
    # a M_PATTERN
    gname = 'seed'
    match = re.search(SEED_PATTERN, suptitle)
    if match is None:
        gname = 'pipeline'
        match = re.search(M_PATTERN, suptitle)
        if match is None:
            return _snakefy(suptitle)
    preffix = suptitle[: (min(match.span()) - 2)]
    group = match.group(1)
    filename = "{0}-{1}{2:02d}".format(_snakefy(preffix), gname, int(group))
    return filename


def metrics_plot(
    metrics: List[float],
    episodes: List[int],
    ylabel: str,
    suptitle: str,
    save_directory_path: Path = None,
    rollouts: bool = True,
) -> None:
    """Plots the `reward`, `|omega|` or any other metric for diagnostics.

    Parameters:
    -----------
    metrics: List[float]
        The metrics collected during training, e.g, rewards.
    episodes: List[int]
        The list with episodes.
    ylabel: str
        The name of the metric.
    suptitle: str
        The title.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.
    rollouts: bool = True
        Saves three rollouts that give an indication of training. Or
        Saves a verylong file of all training steps in sequence.
    """

    metrics = np.array(metrics)
    episodes = np.array(episodes)

    fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    if rollouts:
        Y = []
        first, mid, last = (
            int(np.min(episodes)),
            int(np.mean(episodes)),
            int(np.max(episodes)),
        )
        for idx in (first, mid, last):
            Y.append(metrics[episodes == idx])
        Y = np.vstack(Y).T
    else:
        Y = metrics

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)
    if rollouts:
        labels = tuple(["episode %s" % idx for idx in (first, mid, last)])
        for idx in range(3):
            plt.plot(X, Y[:, idx], label=labels[idx])
    else:
        plt.plot(X, Y, label=ylabel)
        Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)
        plt.plot(*np.hsplit(Y_smooth, indices_or_sections=2), c=SMOOTHING_CURVE_COLOR,label="smooth")


    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend(loc=4)
    plt.suptitle(suptitle)
    _savefig(suptitle, save_directory_path)
    plt.show()


def returns_plot(
    rewards: List[float], episodes: List[int], suptitle, save_directory_path: Path = None
) -> None:
    """Plots the `reward`, `|omega|` or any other metric for diagnostics.

    Parameters:
    -----------
    metrics: List[float]
        The rewards collected during training.
    episodes: List[int]
        The list with episodes.
    suptitle: str
        The title.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.
    """

    rewards = np.array(rewards)
    episodes = np.array(episodes)

    Y = []
    for idx in sorted(set(episodes)):
        Y.append(np.sum(rewards[episodes == idx]))
    Y = np.stack(Y)

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)
    Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X, Y)
    plt.plot(*np.hsplit(Y_smooth, indices_or_sections=2), label="smooth")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward Return")
    plt.legend(loc=4)
    plt.suptitle(suptitle)
    _savefig(suptitle, save_directory_path)
    plt.show()


def train_plot(results: Array, n: int = 1, save_directory_path: Path = None) -> None:
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

    Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X, Y, c="C0", label="mean")
    plt.plot(*np.hsplit(Y_smooth, indices_or_sections=2), c=SMOOTHING_CURVE_COLOR,label="smooth")
    plt.fill_between(X, Y - Y_std, Y + Y_std, facecolor="C0", alpha=0.5)
    plt.suptitle("Train (N=%s, M=%s)" % (n, M))
    plt.xlabel("Episode")
    plt.ylabel("Average Reward Return")
    _savefig("Train Pipeline (M=%s)" % M, save_directory_path)
    plt.show()


def rollout_plot(results: Array, n: int = 1, save_directory_path: Path = None) -> None:
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
    Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X, Y, c="C0")
    plt.plot(*np.hsplit(Y_smooth, indices_or_sections=2), c=SMOOTHING_CURVE_COLOR,label="smooth")
    plt.fill_between(X, Y - Y_std, Y + Y_std, facecolor="C0", alpha=0.5)
    plt.suptitle("Rollouts (N=%s, M=%s)" % (n, M))
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    _savefig("Train Rollout (M=%s)" % M, save_directory_path)
    plt.show()

if __name__ == '__main__':
    #Make gym env
    env = gym.make('CartPole-v1')

    #Run the env
    observation = env.reset()
    frames = []
    for t in range(1000):
        #Render to frames buffer
        frames.append(env.render(mode="rgb_array"))
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    save_frames_as_gif(frames)
