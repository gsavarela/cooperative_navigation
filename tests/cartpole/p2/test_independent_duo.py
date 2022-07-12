"""
BEWARE:
    config.TAU = 1.0

"""
import gym
import numpy as np
import unittest
import matplotlib.pyplot as plt

from independent_learners import ActorCriticIndependent
from tests.cartpole.p2 import DuoCartPoles



class TestCartPole(unittest.TestCase):
    """CartPole task """

    def setUp(self):
        # self.env = gym.make("CartPole-v1")
        self.env = DuoCartPoles()
        self.env.seed((0, 1))

        self.agent = ActorCriticIndependent(
            n_players=2,
            n_features=4,
            action_set=[(0, 0), (1, 0), (0, 1), (1, 1)],
            alpha=0.05,
            beta=0.03,
            zeta=0.01,
            explore_episodes=5000,
            explore=True,
            decay=False,
            seed=0,
        )

    def test_train(self):
        """"""

        returns = []
        first = True
        for _ in range(30000):
            # execution loop
            obs = self.env.reset()
            actions = self.agent.act(obs)
            if not first:
                self.agent.reset()
            else:
                first = False
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

