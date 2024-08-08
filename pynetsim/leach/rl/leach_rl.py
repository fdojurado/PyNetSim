#     PyNetSim: A Python-based Network Simulator for Low-Energy Adaptive Clustering Hierarchy (LEACH) Protocol
#     Copyright (C) 2024  F. Fernando Jurado-Lasso (ffjla@dtu.dk)

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
import pynetsim.leach.surrogate as leach_surrogate


MAX_STEPS = 1000


class LEACH_RL(gym.Env):
    def __init__(self, network, net_model, config):
        super(LEACH_RL, self).__init__()
        self.name = "LEACH_RL"
        self.network = network
        self.net_model = net_model
        self.config = config
        self.episode_network, self.episode_net_model = rl.copy_network(
            network=self.network, net_model=self.net_model)
        self.surrogate_model = SurrogateModel(config=self.config, network=self.episode_network,
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
            protocol=self.surrogate_model, stats=self.stats)

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Instantiate the model
        self.episode_network, self.episode_net_model = rl.copy_network(
            network=self.network, net_model=self.net_model)
        # update the network and net_model of the surrogate model
        self.surrogate_model.update_network(
            network=self.episode_network)
        self.surrogate_model.update_network_model(
            net_model=self.episode_net_model)
        self.surrogate_model.initialize()
        # we step the model one round to get the initial observation
        self.round_number = self.surrogate_model.evaluate_round(round=0)
        self._same_cluster_heads = 0
        self.action = 0
        obs = rl.get_obs(network=self.episode_network, config=self.config,
                         prev_action=self.prev_action, same_cluster_heads=self._same_cluster_heads)
        self.stats = {}
        # input(f"Initial observation: {obs}")
        # print all energy levels
        # for node in self.episode_network:
        #     print(
        #         f"Node {node.node_id} energy: {node.remaining_energy}")

        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
