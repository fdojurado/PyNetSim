import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import copy
import pynetsim.leach.rl as rl
# from pynetsim.network.network import Network
# from pynetsim.config import NETWORK_MODELS
from pynetsim.leach.surrogate.surrogate import SurrogateModel
from pynetsim.leach.leach_milp.leach_ce_e import LEACH_CE_E
import pynetsim.leach.surrogate as leach_surrogate

from pynetsim.utils import PyNetSimLogger

logger_utility = PyNetSimLogger(log_file="my_log.log", namespace=__name__)
logger = logger_utility.get_logger()

MAX_STEPS = 1000


class LEACH_RL_MILP(gym.Env):
    def __init__(self, network, net_model, config, **kwargs):
        super(LEACH_RL_MILP, self).__init__()
        self.name = "LEACH_RL_MILP"
        self.network = network
        self.net_model = net_model
        self.config = config
        # see if we are testing the model
        if 'test' in kwargs:
            logger.info(f"Testing the model")
            test = kwargs['test']
            # if it is not a boolean, raise an error
            if not isinstance(test, bool):
                raise ValueError(
                    f"test must be a boolean, got {type(test)} instead")
            self.test = test
        else:
            self.test = False
        self.episode_network, self.episode_net_model = rl.copy_network(
            network=self.network, net_model=self.net_model)
        self.milp_model = LEACH_CE_E(network=self.episode_network,
                                     net_model=self.episode_net_model)

        self._action = 0
        self.max_steps, self.action, self._same_cluster_heads, self._round_number, self.action_space, self.observation_space = rl.initialize()

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, value):
        self._prev_action = self._action
        self._action = value

    @property
    def prev_action(self):
        return self._prev_action

    @property
    def round_number(self):
        return self._round_number

    @round_number.setter
    def round_number(self, value):
        self._round_number = value

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def net_model(self):
        return self._net_model

    @net_model.setter
    def net_model(self, value):
        self._net_model = value

    def step(self, action):
        self.action, self._same_cluster_heads, self.round_number, obs, reward, done, truncated, info = rl.step(
            action=action, network=self.episode_network, net_model=self.episode_net_model, config=self.config,
            prev_action=self.prev_action, same_cluster_heads=self._same_cluster_heads, round_number=self.round_number,
            protocol=self.milp_model, stats=self.stats, testing=self.test)

        return obs, reward, done, truncated, {'network': self.episode_network,
                                              'network_model': self.episode_net_model}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Instantiate the model
        self.episode_network, self.episode_net_model = rl.copy_network(
            network=self.network, net_model=self.net_model)
        self.milp_model = LEACH_CE_E(network=self.episode_network,
                                     net_model=self.episode_net_model,
                                     alpha=54.82876630831832,
                                     beta=14.53707859358856,
                                     gamma=35.31010127750784)
        for node in self.episode_network:
            node.is_cluster_head = False
            node.dst_to_sink = self.episode_network.distance_to_sink(node)

        # lets set the first cluster heads
        self.round_number = self.milp_model.evaluate_round(round=0)

        self._same_cluster_heads = 0
        self.action = 0
        obs = rl.get_obs(network=self.episode_network, config=self.config,
                         prev_action=self.prev_action, same_cluster_heads=self._same_cluster_heads)
        self.stats = {}
        # input(f"Init observations: {obs}")
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
