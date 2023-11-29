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


MAX_STEPS = 1000


class LEACH_RL_MILP(gym.Env):
    def __init__(self, network, net_model, config):
        super(LEACH_RL_MILP, self).__init__()
        self.name = "LEACH_RL_MILP"
        self.network = network
        self.net_model = net_model
        self.config = config
        self.episode_network, self.episode_net_model = self.copy_network()
        self.milp_model = LEACH_CE_E(network=self.episode_network,
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
        # - number of times the current cluster heads have been cluster heads (int)
        # - Membership of each node to a cluster (array of 99)
        # - previous action (int)
        n_obs = 1+99+5+1+99+1
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(n_obs,), dtype=np.float32)

    def get_remaining_energy(self):
        return self.episode_network.remaining_energy()/50

    def get_energy_consumption(self):
        energy = []
        for node in self.episode_network:
            if node.node_id <= 1:
                continue
            energy.append(node.remaining_energy/0.5)
        return energy

    def get_cluster_heads(self):
        cluster_heads = []
        for node in self.episode_network:
            if node.node_id <= 1:
                continue
            if node.is_cluster_head:
                cluster_heads.append(node.node_id/100)
        # sort
        if len(cluster_heads) < 5:
            cluster_heads.extend([0]*(5-len(cluster_heads)))
        cluster_heads.sort()
        return cluster_heads

    def get_expected_num_cluster_heads(self):
        num_alive_nodes = self.episode_network.alive_nodes()
        # percentage of cluster heads
        p = self.config.network.protocol.cluster_head_percentage
        expected_num_cluster_heads = int(num_alive_nodes * p)+1
        return expected_num_cluster_heads

    def get_membership(self):
        membership = []
        for node in self.episode_network:
            if node.node_id <= 1:
                continue
            if node.cluster_id is None:
                print(f"Node {node.node_id} has no cluster id")
                print(f"Node energy: {node.remaining_energy}")
                print(f"alive nodes: {self.episode_network.alive_nodes()}")
                membership.append(0)
            else:
                membership.append(node.cluster_id/100)
        return membership

    def _get_obs(self):
        re = self.get_remaining_energy()
        assert re >= -1 and re <= 1, f"Remaining energy: {re}"
        nodes_energy_consumption = self.get_energy_consumption()
        assert len(
            nodes_energy_consumption) == 99, f"Length of nodes energy consumption: {len(nodes_energy_consumption)}"
        assert all(
            [x >= -1 and x <= 1 for x in nodes_energy_consumption]), f"Nodes energy consumption: {nodes_energy_consumption}"
        current_cluster_heads = self.get_cluster_heads()
        # expected_num_cluster_heads = self.get_expected_num_cluster_heads()
        # assert len(
        #     current_cluster_heads) == expected_num_cluster_heads, f"Length of current cluster heads: {len(current_cluster_heads)}, expected: {expected_num_cluster_heads}"
        # assert all(
        #     [x >= 0 and x <= 1 for x in current_cluster_heads]), f"Current cluster heads: {current_cluster_heads}"
        num_times_ch = self._same_cluster_heads/400
        assert num_times_ch >= 0 and num_times_ch <= 1, f"Number of times CH: {num_times_ch}"
        membership = self.get_membership()
        assert len(
            membership) == 99, f"Length of membership: {len(membership)}"
        assert all(
            [x >= 0 and x <= 1 for x in membership]), f"Membership: {membership}"
        prev_action = self._prev_action
        assert prev_action >= 0 and prev_action <= 1, f"Previous action: {prev_action}"
        return np.array([re, *nodes_energy_consumption, *current_cluster_heads, num_times_ch, *membership, prev_action])

    def copy_network(self):
        network_copy = copy.deepcopy(self.network)
        net_model_copy = copy.deepcopy(self.net_model)
        network_copy.set_model(net_model_copy)
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
        reward = 0

        # Two actions, create a new set of clusters or stay in the same set
        if self.action == 0:
            self._same_cluster_heads = 0
            # print("Stay in the same set of clusters")
            self.round_number += 1
            self.episode_net_model.dissipate_energy(round=self.round_number)
            # reward = 0.5
        if self.action == 1:
            self._same_cluster_heads += 1
            # print("Create a new set of clusters")
            self.round_number = self.milp_model.evaluate_round(
                round=self.round_number)
            reward += 0.1

        # Are there any dead nodes?
        alive_nodes = self.episode_network.alive_nodes()
        if alive_nodes <= 0:
            done = True
            reward = 2
        else:
            reward += 1

        obs = self._get_obs()

        return obs, reward, done, False, {'network': self.episode_network,
                                          'network_model': self.episode_net_model}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Instantiate the model
        self.episode_network, self.episode_net_model = self.copy_network()
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
        self._prev_action = 0
        self._action = 0
        obs = self._get_obs()
        # input(f"Init observations: {obs}")
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
