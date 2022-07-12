import gym

class CartPoles(object):
    """Wrapper class for cartpole"""
    def __init__(self, n_players=2):
        self.envs = [gym.make("CartPole-v1") for _ in range(n_players)]

    def reset(self):
        ret = []
        for _env in self.envs:
            ret.append(_env.reset())
        return ret

    def seed(self, seeds):
        for _env, _seed in zip(self.envs, seeds):
            _env.seed(_seed)

    def step(self, actions):
        # obs, rewards, dones, infos = []
        ret = []
        for _env, _action in zip(self.envs, actions):
            # n_obs, n_rewards, n_done, n_info = _env.step(*_action)
            ret.append(_env.step(_action))
        ret = tuple([*zip(*ret)])
        return ret

