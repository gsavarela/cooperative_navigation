"""Common functions for the project."""
from enum import Enum
from typing import List, Tuple
from itertools import product
from operator import itemgetter

import numpy as np

# Constants
N_ACTIONS = 5  # Per player
N_FEATURES = 4  # Per player

EYE = np.eye(N_ACTIONS)


# Enumerators
class PlayerActions(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
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
