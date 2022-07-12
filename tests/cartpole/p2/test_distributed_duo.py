"""
BEWARE:
    config.TAU = 1.0

"""
import gym
import numpy as np
import unittest
import matplotlib.pyplot as plt

from distributed_learners import ActorCriticDistributedV
from tests.cartpole.p2 import DuoCartPoles

class TestDuoCartPoles(unittest.TestCase):
    """CartPole task """

    def setUp(self):

        self.env = DuoCartPoles()
        self.env.seed((0, 1))

        self.agent = ActorCriticDistributedV(
            n_players=2,
            n_features=8,
            action_set=[(0, 0), (1, 0), (0, 1), (1, 1)],
            alpha=0.5,
            beta=0.3,
            zeta=0.01,
            explore_episodes=2500,
            explore=False,
            decay=False,
            seed=0,
        )

    def test_train(self):
        returns = []
        for i in range(20000):
            # execution loop
            obs = self.env.reset()
            actions = self.agent.act(obs)
            if i > 0:
                self.agent.reset()
            rewards = []
            for _ in range(100):
                # step environment
                next_obs, next_rewards, dones, _ = self.env.step(actions)

                # actor parameters.
                next_actions = self.agent.act(next_obs)

                self.agent.update(obs, actions, next_rewards, next_obs, next_actions)

                obs = next_obs
                actions = next_actions
                rewards.append(np.mean(next_rewards))
                if any(dones):
                    break
            returns.append(np.sum(rewards))
        plt.plot(returns)
        plt.show()
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

