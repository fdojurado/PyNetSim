import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pynetsim.leach.rl as rl
# from pynetsim.config import PyNetSimConfig


class LEACH_RL(gym.Env):
    def __init__(self, network, net_model: object):
        self.name = "LEACH_RL"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.max_steps = self.config.network.protocol.max_steps
        self.action = 0

        self.round = 0

        # Calculate the maximum distance to the sink
        self.max_distance = np.sqrt(
            self.config.network.width**2 + self.config.network.height**2)

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

        self.actions_dict = {}
        count = 0
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            self.actions_dict[count] = node.node_id
            count += 1
        # print(f"Actions: {self.actions_dict}")
        n_actions = len(self.actions_dict)
        # print(f"Action space: {n_actions}")

        # Observation space: energy consumption + cluster head indicators
        n_observation = (6 * (self.config.network.num_sensor+1) + 3)*2
        # print(f"Observation space: {n_observation}")
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_observation,), dtype=np.float32)
        self.prev_obs = np.zeros(int(n_observation/2))
        self.action_space = spaces.Discrete(n_actions)

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
                     action_taken=self.action/len(self.actions_dict))

        # Append the previous observation
        observations = np.append(obs, self.prev_obs)

        # print(f"obs len: {len(obs)}")
        # print(f"Observations len: {len(observations)}")
        # print(f"prev obs len: {len(self.prev_obs)}")

        self.prev_obs = obs
        info = {}

        return observations, info

    def _calculate_reward(self):
        current_energy = self.network.remaining_energy()
        self.net_model.dissipate_energy(round=self.round)
        latest_energy = self.network.remaining_energy()
        energy = (current_energy - latest_energy)*10
        # Check that the energy is between 0 and 1
        assert energy >= 0 and energy <= 1, f"Energy: {energy}"
        reward = 2 - 1 * energy
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.round = 0

        # Create a random network of num_nodes nodes and random positions and
        # random energy levels
        rl.create_network(network=self.network, config=self.config)

        # print all cluster heads
        # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        # ) if cluster_head.is_cluster_head]
        # print(f"Cluster heads: {chs}")

        self.network.create_clusters()

        self.net_model.dissipate_energy(round=self.round)

        observation, info = self._get_obs()

        return observation, info

    def step(self, action):
        self.round += 1
        self.action = int(action)
        action = self.actions_dict[int(action)]
        node = self.network.nodes[action]
        done = False

        if node.energy < self.network.average_energy():
            obs, info = self._get_obs()
            return obs, 0, True, False, info

        if node.is_cluster_head:
            self.network.mark_as_non_cluster_head(node)
        else:
            self.network.mark_as_cluster_head(node, node.node_id)

        self.network.create_clusters()
        reward = self._calculate_reward()
        observation, info = self._get_obs()
        return observation, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
