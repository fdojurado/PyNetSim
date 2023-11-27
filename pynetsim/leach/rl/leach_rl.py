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
        self.episode_network, self.episode_net_model = self.copy_network()
        self.surrogate_model = SurrogateModel(config=self.config, network=self.episode_network,
                                              net_model=self.episode_net_model)

        self.max_steps = MAX_STEPS
        self._action = 0
        self._prev_action = 0
        self._same_cluster_heads = 0

        self._round_number = 0

        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # Observation are:
        # - Remaining energy of the network (float)
        # - Energy consumption of each node (array of 99)
        # - current cluster heads (array of 5)
        # - previous cluster heads (array of 5)
        # - number of times the current cluster heads have been cluster heads (int)
        # - Membership of each node to a cluster (array of 99)
        # - Number of alive nodes (int)
        # - Current round (int)
        # - Expected energy consumption of each node for both cluster heads and non cluster heads (array of 99)
        # - previous action (int)
        n_obs = 99 + 5 + 5 + 1 + 99 + 1 + 1 + 1 + 99 + 1
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(n_obs,), dtype=np.float32)

    def _get_obs(self):
        re = self.episode_network.remaining_energy()
        std_re = leach_surrogate.standardize_inputs(
            x=re, mean=self.surrogate_model.cluster_head_model.re_mean,
            std=self.surrogate_model.cluster_head_model.re_std)
        std_el = leach_surrogate.get_standardized_energy_levels(
            network=self.network,
            mean=self.surrogate_model.cluster_head_model.el_mean,
            std=self.surrogate_model.cluster_head_model.el_std)
        current_cluster_heads = self.episode_network.get_cluster_head_ids_at_round(
            self.round_number)
        if self.round_number - 1 <= 0:
            prev_cluster_heads = [0, 0, 0, 0, 0]
        else:
            prev_cluster_heads = self.episode_network.get_cluster_head_ids_at_round(
                self.round_number - 1)
        num_times_ch = self._same_cluster_heads
        memb = [node.cluster_id for node in self.episode_network if node.node_id != 1]
        alive_nodes = self.episode_network.alive_nodes()
        round_number = self.round_number
        expected_energy_consumption = []
        for node in self.episode_network:
            if node.node_id <= 1:
                continue
            if node.is_cluster_head:
                energy = self.surrogate_model.cluster_head_model.estimate_tx_energy[
                    node.node_id][1]
            else:
                energy = self.surrogate_model.cluster_head_model.estimate_tx_energy[
                    node.node_id][node.cluster_id]
            expected_energy_consumption.append(energy)
        expected_energy_consumption = leach_surrogate.standardize_inputs(
            x=expected_energy_consumption,
            mean=self.surrogate_model.cluster_head_model.el_mean,
            std=self.surrogate_model.cluster_head_model.el_std)
        prev_action = self.prev_action
        return np.array([std_re, *std_el, *current_cluster_heads, *prev_cluster_heads, num_times_ch, *memb, alive_nodes, round_number, *expected_energy_consumption, prev_action])

    def copy_network(self):
        network_copy = copy.deepcopy(self.network)
        net_model_copy = copy.deepcopy(self.net_model)
        net_model_copy.set_network(network_copy)
        # Register callback to the network model
        # self.model.register_round_complete_callback(self.round_callback)
        # Register the callback to the network
        net_model_copy.register_round_complete_callback(
            network_copy.round_callback)
        return network_copy, net_model_copy

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

    @property
    def std_re(self):
        return self._std_re

    @std_re.setter
    def std_re(self, value):
        self._std_re = value

    @property
    def avg_re(self):
        return self._avg_re

    @avg_re.setter
    def avg_re(self, value):
        self._avg_re = value

    def _calculate_reward(self):
        pass

    def step(self, action):
        self.action = int(action)
        done = False

        # Two actions, create a new set of clusters or stay in the same set
        if self.action == 0:
            pass
            print("Stay in the same set of clusters")
        if self.action == 1:
            self._same_cluster_heads += 1
            print("Create a new set of clusters")

        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Instantiate the model
        self.episode_network, self.episode_net_model = self.copy_network()
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
        obs = self._get_obs()

        print(f"Initial observation: {obs}")

        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
