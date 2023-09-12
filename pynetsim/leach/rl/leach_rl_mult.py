from gymnasium import spaces

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import pynetsim.leach.rl as rl
import copy
# from pynetsim.config import PyNetSimConfig


class LEACH_RL_MULT(gym.Env):
    def __init__(self, network, net_model: object):
        self.name = "LEACH_RL_MULT"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.max_steps = self.config.network.protocol.max_steps
        self.action = 0

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

    def set_network(self, network):
        self.network = network

    def set_net_model(self, net_model):
        self.net_model = net_model
        self.net_model.set_network(self.network)

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

    def _update_cluster_heads(self):
        for node in self.network:
            if node.node_id == 1:
                continue
            node_copy = self.network_copy.get_node(node.node_id)
            node.is_cluster_head = node_copy.is_cluster_head
            node.cluster_id = node_copy.cluster_id

    def _print_network_info(self, msg, network=None):
        # Print cluster heads
        chs = [
            cluster_head.node_id for cluster_head in network if cluster_head.is_cluster_head]
        print(f"{msg} Cluster heads: {chs}")
        # print sensor nodes' remaining energy
        for node in network:
            if node.node_id == 1:
                continue
            print(f"{msg} Node {node.node_id}: {node.remaining_energy}")

    def _calculate_reward(self):
        current_energy = self.network_copy.remaining_energy()
        # print(f"Current energy: {current_energy}")
        self.net_model_copy.dissipate_energy(round=self.round)
        latest_energy = self.network_copy.remaining_energy()
        # print(f"Latest energy: {latest_energy}")
        energy = (current_energy - latest_energy)*10
        # print(f"Energy: {energy}")
        # Check that the energy is between 0 and 1
        assert energy >= 0 and energy <= 1, f"Energy: {energy}"
        reward = 2 - 1 * energy
        return reward

    def reset(self, seed=None, options: dict = None):
        super().reset(seed=seed)
        self.round = 0

        # if the options are not provided, then create a random network
        # otherwise, create a network based on the options.
        # The options are a dictionary with the following keys:
        #   - num_nodes: number of nodes in the network
        #   - num_chs: number of cluster heads in the network

        if options is None:
            # Create a random network of num_nodes nodes and random positions and
            # random energy levels
            rl.create_network(network=self.network, config=self.config,
                              lower_energy=0)

            # print all cluster heads
            # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
            # ) if cluster_head.is_cluster_head]
            # print(f"Cluster heads at reset: {chs}")

            # self.network.create_clusters()

            # self._print_network_info("Reset:", self.network)
        else:
            # Mark all nodes as non-cluster heads
            for node in self.network:
                self.network.mark_as_non_cluster_head(node)
            # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
            # ) if cluster_head.is_cluster_head]
            # print(f"Cluster heads at reset: {chs}")

        observation, info = self._get_obs()

        return observation, info

    def step(self, action):
        self.round += 1
        # Network average remaining energy
        network_avg_energy = self.network.average_remaining_energy()
        # Make a deep copy of the network
        self.network_copy = copy.deepcopy(self.network)
        # Make a deep copy of the network model
        self.net_model_copy = copy.deepcopy(self.net_model)
        self.net_model_copy.set_network(self.network_copy)
        self.action = int(action)
        action = self.actions_dict[int(action)]
        node = self.network_copy.get_node(action)
        done = False
        penalty = False
        # print(f"Network average remaining energy: {network_avg_energy}")
        # print(f"Node {node.node_id} remaining energy: {node.remaining_energy}")

        if node.node_id in self.network.get_cluster_head_ids():
            # obs, info = self._get_obs()
            # print(
            #     f"LL: Penalty for selecting node {node.node_id} as it is already a CH")
            penalty = True
            # return obs, -10, True, False, info
        if node.remaining_energy < network_avg_energy:
            # obs, info = self._get_obs()
            # print(f"LL: Penalty for selecting node {node.node_id} as CH")
            penalty = True
            # return obs, -10, True, False, info
        # print(f"Node {node.node_id} is not a cluster head, mark as CH")
        self.network_copy.mark_as_cluster_head(node, node.node_id)
        # print(
        #     f"Node {node.node_id} is a cluster head with cluster id {node.cluster_id}")

        self.network_copy.create_clusters()
        reward = self._calculate_reward()
        # Update the network clusters and network model
        self._update_cluster_heads()
        # print network information
        # self._print_network_info("Step:", self.network_copy)
        # print(f"Reward: {reward}")

        observation, info = self._get_obs()

        if penalty:
            # input(f"Penalty for selecting node {node.node_id} as CH")
            reward = -10
            return observation, reward, True, False, info

        # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        # ) if cluster_head.is_cluster_head]
        # The episode is over when the number of cluster heads is equal to the
        # number of alive nodes times the percentage of cluster heads
        if self.network.num_cluster_heads() >= (self.network.alive_nodes() *
                                                self.config.network.protocol.cluster_head_percentage):
            done = True
            # print(f"Episode is over at round {self.round}")
            reward += 3
        # input(f"Cluster heads at round {self.round}: {chs}")

        return observation, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
