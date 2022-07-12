"""
BEWARE:
    config.TAU = 5.0

"""
import gym
import numpy as np
import unittest
import matplotlib.pyplot as plt

import config
from central import ActorCriticCentral



class TestCartPole(unittest.TestCase):
    """CartPole task """

    def setUp(self):
        self.env = gym.make("CartPole-v1")
        self.env.seed(0)

        self.agent = ActorCriticCentral(
        n_players=1,
        n_features=4,
        action_set=[(0,), (1,)],
        alpha=0.2,
        beta=0.1,
        zeta=0.01,
        explore_episodes=2500,
        explore=False,
        decay=False,
        seed=0,
    )

    def test_train(self):
        """"""

        assert config.TAU == 5.0, "Set config.TAU to 5.0 before executing this simulation"
        returns = []
        for _ in range(5000):
            # execution loop
            obs = self.env.reset()
            actions = self.agent.act(obs)
            rewards = []
            for _ in range(100):
                # step environment
                next_obs, next_rewards, done, _ = self.env.step(*actions)

                # actor parameters.
                next_actions = self.agent.act(next_obs)

                self.agent.update(obs, actions, next_rewards, next_obs, next_actions)

                obs = next_obs
                actions = next_actions
                rewards.append(np.mean(next_rewards))
                if done:
                    break
            returns.append(np.sum(rewards))
        plt.plot(returns)
        plt.show()
        

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
