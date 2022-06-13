"""Common functions for the project."""
from collections.abc import Iterable
from enum import Enum
from typing import List, Tuple
from itertools import product
from operator import itemgetter
from pathlib import Path
import string

import numpy as np

# Constants
N_ACTIONS = 5  # Per player
N_FEATURES = 4  # Per player

EYE = np.eye(N_ACTIONS)


# Enumerators
class PlayerActions(Enum):
    NONE = 0
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Types
Array = np.ndarray
Observation = List[Array]
Action = Tuple[int]
ActionSet = List[Action]
ActionsOneHot = Array
Rewards = Array
Step = Tuple[Observation, Array]


# Functions
def softmax(x: Array) -> Array:
    expx = np.exp(x - np.max(x))
    return expx / expx.sum(keepdims=True)


def action_set(n_agents: int = 1) -> ActionSet:
    res = product(np.arange(N_ACTIONS).tolist(), repeat=n_agents)
    for i in range(n_agents - 1, -1, -1):
        res = sorted(res, key=itemgetter(i))
    return [*res]


def onehot(action: Action) -> ActionsOneHot:
    return EYE[action, :]


def snakefy(title_case: str) -> str:
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

def flatten(items, ignore_types=(str, bytes)):
    """Gets a mixed list of elements and other lists

    Usage:
    -----
    > items = [1, 2, [3, 4, [5, 6], 7], 8]

    > # Produces 1 2 3 4 5 6 7 8
    > for x in flatten(items):
    >         print(x)

    Ref:
    ----

    David Beazley. `Python Cookbook.'
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x

def make_dirs(save_directory_path: Path) -> None:
    # iteractively check for path and builds where it doesnt exist.
    prev_path = None
    for sub_dir in save_directory_path.parts:
        if prev_path is None:
            prev_path = Path(sub_dir)
        else:
            prev_path = prev_path / sub_dir
        prev_path.mkdir(exist_ok=True)
