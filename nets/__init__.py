from typing import Tuple
import torch
from torch import nn
from torch.distributions import Categorical
import torch.optim as optim

from common import Array
from nets.critic import CriticNetwork
from nets.actor import ActorNetwork


def select_action(anet: nn.Module, state: Array) -> Tuple[int, float]:
    """Performs a feed-forward operation on the network selecting an action

    Parameters
    ----------
    * anet: nn.Module
        The actor neural network
    * state: Array
        Array of action space in an environment

    Returns
    -------
    * action: int
        The selected action
    * log_prob:
        log probability of selecting that action given state and network
    """
    # convert state to float tensor, add 1 dimension, allocate tensor on device
    state = torch.from_numpy(state).float().unsqueeze(0).to("cpu")

    # use network to predict action probabilities
    action_probs = anet(state)
    state = state.detach()

    # sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()

    # return action
    return action.item(), m.log_prob(action)


def get_optimizer(net: nn.Module, lr: float):
    return optim.SGD(net.parameters(), lr=lr)

__all__ = [ActorNetwork, CriticNetwork, get_optimizer, select_action]
