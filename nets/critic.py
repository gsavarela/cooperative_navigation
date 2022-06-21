"""The critic approximation with stocastic gradient descent"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Using a neural network to learn state value
class CriticNetwork(nn.Module):
    # Takes in state
    def __init__(self, observation_space: int, lr: float = 0.01):
        super(CriticNetwork, self).__init__()

        self.input_layer = nn.Linear(observation_space, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        # input layer
        x = self.input_layer(x)

        # activiation relu
        x = F.relu(x)

        # get state value
        state_value = self.output_layer(x)

        return state_value
