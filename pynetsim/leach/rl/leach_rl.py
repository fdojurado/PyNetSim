import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pynetsim.leach.rl as rl
import math
# from pynetsim.config import PyNetSimConfig
from pynetsim.leach.leach_k import LEACH_K
import pynetsim.leach.leach_milp as leach_milp


# Define the rewards and penalties
REWARD_ROUND_NO_DEATHS = 1  # Positive reward for rounds without deaths
PENALTY_NODE_DEATH = -1  # Penalty for node death
PENALTY_NODE_FACTOR = -100  # Penalty for node death
PENALTY_NODE_EXPONENT = 0.1  # Penalty exponent for node death
EXPLORATION_BONUS = 0.1  # Small positive bonus for exploration
TERMINATION_BONUS_FACTOR = 15  # Bonus for prolonging the episode


class LEACH_RL(gym.Env):
    def __init__(self, network, net_model: object):
        self.name = "LEACH_RL"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.max_steps = self.config.network.protocol.max_steps
        self.action = 0

        self.first_dead_node = False

        self.round = 0

        # Calculate the maximum distance to the sink, which is the distance
        # between the sink and the farthest node
        self.max_distance = self.network.max_distance_to_sink()

        self.dst_to_sink = np.zeros(self.config.network.num_sensor+1)
        # Set all dst_to_sink for all nodes
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            node.dst_to_sink = self.network.distance_to_sink(node)
            self.dst_to_sink[node.node_id] = node.dst_to_sink/self.max_distance

        # Initialize array of nodes' positions
        self.x_locations = np.zeros(self.config.network.num_sensor+1)
        self.y_locations = np.zeros(self.config.network.num_sensor+1)
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            self.x_locations[node.node_id] = node.x/self.config.network.width
            self.y_locations[node.node_id] = node.y/self.config.network.height

        n_actions = 2
        # print(f"Action space: {n_actions}")

        # Observation space: energy consumption + cluster head indicators
        n_observation = (6 * (self.config.network.num_sensor+1) + 3)*2
        # print(f"Observation space: {n_observation}")
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_observation,), dtype=np.float32)
        self.prev_obs = np.zeros(int(n_observation/2))
        self.action_space = spaces.Discrete(n_actions)

    def set_network(self, network):
        self.network = network

    def set_net_model(self, net_model):
        self.net_model = net_model

    def _get_obs(self):
        obs = rl.obs(num_sensors=self.config.network.num_sensor,
                     network=self.network,
                     x_pos=self.x_locations,
                     y_pos=self.y_locations,
                     dst_to_sink=self.dst_to_sink,
                     init_energy=self.config.network.protocol.init_energy,
                     round=self.round,
                     max_steps=self.max_steps,
                     max_distance=self.max_distance,
                     action_taken=self.action/2)

        # Append the previous observation
        observations = np.append(obs, self.prev_obs)

        # print(f"obs len: {len(obs)}")
        # print(f"Observations len: {len(observations)}")
        # print(f"prev obs len: {len(self.prev_obs)}")

        self.prev_obs = obs
        info = {}

        return observations, info

    def _calculate_reward(self):
        self.net_model.dissipate_energy(round=self.round)
        nodes_died = self.prev_alive_nodes - self.network.alive_nodes()

        # Calculate the reward
        reward = 0

        if self.action == 1:
            reward += EXPLORATION_BONUS

        if nodes_died > 0:
            if not self.first_dead_node:
                reward += PENALTY_NODE_DEATH * 50
                self.first_dead_node = True
            else:
                reward += PENALTY_NODE_DEATH
        else:
            if self.first_dead_node:
                reward += REWARD_ROUND_NO_DEATHS * 0.01
            else:
                reward += REWARD_ROUND_NO_DEATHS

        # Previous alive nodes
        self.prev_alive_nodes = self.network.alive_nodes()

        return reward

        # self.net_model.dissipate_energy(round=self.round)
        # nodes_died = self.prev_alive_nodes - self.network.alive_nodes()

        # # Calculate the reward
        # reward = 0

        # if nodes_died == 0:
        #     reward += REWARD_ROUND_NO_DEATHS
        # else:
        #     reward += PENALTY_NODE_DEATH

        # # Encourage exploration with an exploration bonus
        # reward += EXPLORATION_BONUS

        # # Previous alive nodes
        # self.prev_alive_nodes = self.network.alive_nodes()

        # # Episode termination bonus (encourage the agent to prolong the episode)
        # if self.prev_alive_nodes == 0:
        #     # All nodes are dead, end the episode
        #     termination_bonus = self.round
        #     reward += termination_bonus

        # return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.round = 0
        self.new_cluster = 0
        self.first_dead_node = False

        self.prev_alive_nodes = self.network.alive_nodes()

        # Create a random network of num_nodes nodes and random positions and
        # random energy levels
        rl.create_clustered_network(network=self.network, config=self.config)

        # print all cluster heads
        # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        # ) if cluster_head.is_cluster_head]
        # print(f"Cluster heads at reset: {chs}")

        # self.network.create_clusters()

        # self.net_model.dissipate_energy(round=self.round)

        observation, info = self._get_obs()

        return observation, info

    def step(self, action):
        self.round += 1
        self.action = int(action)
        done = False

        # Two actions, create a new set of clusters or stay in the same set
        if self.action == 0:
            pass
            # print("Stay in the same set of clusters")
        if self.action == 1:
            # Mark all nodes as non cluster heads
            for node in self.network:
                self.network.mark_as_non_cluster_head(node)
            # print(f"Create a new set of clusters at {self.new_cluster} round {self.round}!")
            LEACH_K.choose_cluster_heads(
                network=self.network, config=self.config)

        reward = self._calculate_reward()
        observation, info = self._get_obs()
        # We end the episode when there are not alive nodes
        if self.network.alive_nodes() == 0:
            done = True
            # input(f"Dead node at round {self.round}")
        # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        # ) if cluster_head.is_cluster_head]
        # input(f"Cluster heads at round {self.round}: {chs}")
        return observation, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
