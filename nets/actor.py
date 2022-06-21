"""The actor approximation with stocastic gradient descent"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Using a neural network to learn our policy parameters
class ActorNetwork(nn.Module):
    # Takes in observations and outputs actions
    def __init__(self, observation_space: int, action_space: int, lr: float = 0.001):
        super(ActorNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 32)
        self.output_layer = nn.Linear(32, action_space)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)

        # relu activation
        x = F.relu(x)

        # actions
        actions = self.output_layer(x)

        # get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)

        return action_probs
