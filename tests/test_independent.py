"""
BEWARE:
    config.TAU = 1.0

"""
import gym
import numpy as np
import unittest
import matplotlib.pyplot as plt

from independent_learners import ActorCriticIndependent



class TestCartPole(unittest.TestCase):
    """CartPole task 

    Set tau = 1.0
    """

    def setUp(self):
        self.env = gym.make("CartPole-v1")
        self.env.seed(0)

        self.agent = ActorCriticIndependent(
        n_players=1,
        n_features=4,
        action_set=[(0,), (1,)],
        alpha=0.5,
        beta=0.3,
        zeta=0.01,
        explore_episodes=1000,
        explore=False,
        decay=False,
        seed=0,
    )

    def test_train(self):
        """"""

        returns = []
        for _ in range(5000):
            # execution loop
            obs = self.env.reset()
            actions = self.agent.act([obs])
            rewards = []
            for _ in range(100):
                # step environment
                next_obs, next_rewards, done, _ = self.env.step(*actions)

                # actor parameters.
                next_actions = self.agent.act([next_obs])

                self.agent.update([obs], actions, [next_rewards], [next_obs], next_actions)

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


# class TestGridAveragePressureNorm(TestGridAveragePressure):
#     """
#         * Tests average pressure related state and reward
#
#         * Normalize state space by number of vehicles
#
#         * Set of tests that target the implemented
#           problem formulations, i.e. state and reward
#           function definitions.
#
#         * Use lazy_properties to compute once and use
#           as many times as you want -- it's a cached
#           property
#     """
#     @property
#     def mdp_params(self):
#         mdp_params = MDPParams(
#                         features=('average_pressure',),
#                         reward='reward_min_average_pressure',
#                         normalize_velocities=True,
#                         normalize_vehicles=self.norm_vehs,
#                         discretize_state_space=False,
#                         reward_rescale=0.01,
#                         time_period=None,
#                         velocity_threshold=0.1)
#         return mdp_params
#
#     @property
#     def norm_vehs(self):
#         return True
#
#     def test_avg_pressure_tl1ph0(self):
#         """Tests avg.pressure state
#             * traffic light 1
#             * ID = '247123161'
#             * phase 0
#         """
#         ID = '247123161'
#
#         outgoing = OUTGOING_247123161
#         incoming = INCOMING_247123161[0]
#         fct1 = MAX_VEHS[(ID, 0)] if self.norm_vehs else 1
#         fct2 = MAX_VEHS_OUT[(ID, 0)] if self.norm_vehs else 1
#
#         p0 = process_pressure(self.kernel_data_1, incoming, outgoing,
#                               fctin=fct1, fctout=fct2, is_average=True)
#
#         # State.
#         # 247123161 static assertion
#         self.assertEqual(self.state[ID][0], 0.05) # pressure, phase 0
#
#         # 247123161 dynamic assertion
#         self.assertEqual(self.state[ID][0], p0) # pressure, phase 0
#
#     def test_avg_pressure_tl1ph1(self):
#         """Tests avg.pressure state
#             * traffic light 1
#             * ID = '247123161'
#             * phase 1
#         """
#         ID = '247123161'
#
#         outgoing = OUTGOING_247123161
#         incoming = INCOMING_247123161[1]
#         fct1 = MAX_VEHS[(ID, 1)] if self.norm_vehs else 1
#         fct2 = MAX_VEHS_OUT[(ID, 1)] if self.norm_vehs else 1
#
#         p1 = process_pressure(self.kernel_data_1, incoming, outgoing,
#                               fctin=fct1, fctout=fct2, is_average=True)
#
#         # State.
#         # 247123161 static assertion
#         self.assertEqual(self.state[ID][1], 0.02) # pressure, phase 1
#         # 247123161 dynamic assertion
#         self.assertEqual(self.state[ID][1], p1) # pressure, phase 1
#
#
#     def test_min_avg_pressure_tl1(self):
#         """Tests pressure reward
#             * traffic light 1
#             * ID = '247123161'
#         """
#         ID = '247123161'
#         reward = self.reward(self.observation_space)
#         self.assertEqual(reward[ID], round(-0.01*(0.05 + 0.02), 4))
#
#
#     def test_avg_pressure_tl2ph0(self):
#         """Tests pressure state
#             * traffic light 2
#             * ID = '247123464'
#             * phase 0
#         """
#         ID = '247123464'
#
#         outgoing = OUTGOING_247123464
#         incoming = INCOMING_247123464[0]
#         fct1 = MAX_VEHS[(ID, 0)] if self.norm_vehs else 1
#         fct2 = MAX_VEHS_OUT[(ID, 0)] if self.norm_vehs else 1
#
#         p0 = process_pressure(self.kernel_data_1, incoming, outgoing,
#                               fctin=fct1, fctout=fct2, is_average=True)
#
#         # State.
#         # 247123464 static assertion
#         self.assertEqual(self.state[ID][0], -0.05) # pressure, phase 0
#
#         # 247123464 dynamic assertion
#         self.assertEqual(self.state[ID][0], p0) # pressure, phase 0
#
#     def test_avg_pressure_tl2ph1(self):
#         """Tests pressure state
#             * traffic light 2
#             * ID = '247123464'
#             * phase 1
#         """
#         ID = '247123464'
#
#         outgoing = OUTGOING_247123464
#         incoming = INCOMING_247123464[1]
#
#         fct1 = MAX_VEHS[(ID, 1)] if self.norm_vehs else 1
#         fct2 = MAX_VEHS_OUT[(ID, 1)] if self.norm_vehs else 1
#
#         p1 = process_pressure(self.kernel_data_1, incoming, outgoing,
#                               fctin=fct1, fctout=fct2, is_average=True)
#
#         # State.
#         # 247123464 static assertion
#         self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1
#
#         # 247123464 dynamic assertion
#         self.assertEqual(self.state[ID][1], p1) # pressure, phase 1
#
#
#     def test_min_avg_pressure_tl2(self):
#         """Tests pressure reward
#             * traffic light 2
#             * ID = '247123464'
#         """
#         ID = '247123464'
#         reward = self.reward(self.observation_space)
#         self.assertAlmostEqual(reward[ID], round(-0.01*(-0.05 - 0.0), 4))
#
#     def test_avg_pressure_tl3ph0(self):
#         """Tests pressure state
#             * traffic light 3
#             * ID = '247123468'
#             * phase 0
#         """
#         ID = '247123468'
#
#         outgoing = OUTGOING_247123468
#         incoming = INCOMING_247123468[0]
#
#         fct1 = MAX_VEHS[(ID, 0)] if self.norm_vehs else 1
#         fct2 = MAX_VEHS_OUT[(ID, 0)] if self.norm_vehs else 1
#
#         p0 = process_pressure(self.kernel_data_1, incoming, outgoing,
#                               fctin=fct1, fctout=fct2, is_average=True)
#
#         # State.
#         # 247123468 static assertion
#         self.assertEqual(self.state[ID][0], 0.07) # pressure, phase 0
#
#         # 247123468 dynamic assertion
#         self.assertEqual(self.state[ID][0], p0) # pressure, phase 0
#
#
#     def test_avg_pressure_tl3ph1(self):
#         """Tests pressure state
#             * traffic light 3
#             * ID = '247123468'
#             * phase 1
#         """
#         ID = '247123468'
#
#         outgoing = OUTGOING_247123468
#         incoming = INCOMING_247123468[1]
#         fct1 = MAX_VEHS[(ID, 1)] if self.norm_vehs else 1
#         fct2 = MAX_VEHS_OUT[(ID, 1)] if self.norm_vehs else 1
#
#         p1 = process_pressure(self.kernel_data_1, incoming, outgoing,
#                               fctin=fct1, fctout=fct2, is_average=True)
#
#         # State.
#         # 247123468 static assertion
#         self.assertEqual(self.state[ID][1], 0.01) # pressure, phase 1
#
#         # 247123468 dynamic assertion
#         self.assertEqual(self.state[ID][1], p1) # pressure, phase 1
#
#     def test_min_avg_pressure_tl3(self):
#         """Tests pressure reward
#             * traffic light 3
#             * ID = '247123468'
#         """
#         ID = '247123468'
#         reward = self.reward(self.observation_space)
#         self.assertEqual(reward[ID], round(-0.01*(0.07 + 0.01), 4))

if __name__ == '__main__':
    unittest.main()

