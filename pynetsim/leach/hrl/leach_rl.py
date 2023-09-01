import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pynetsim.leach as leach
from pynetsim.node.node import Node
import pynetsim.leach.hrl as hrl
# from pynetsim.config import PyNetSimConfig


class LEACH_RL(gym.Env):
    def __init__(self, network):
        self.name = "LEACH_RM"
        self.config = network.config
        self.network = network
        self.max_steps = self.config.network.protocol.max_steps
        self.action = 0

        # Energy conversion factors
        self.elect, self.etx, self.erx, self.eamp, self.eda, self.packet_size = leach.get_energy_conversion_factors(
            self.config)

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
        print(f"Actions: {self.actions_dict}")
        n_actions = len(self.actions_dict)
        print(f"Action space: {n_actions}")

        # Observation space: energy consumption + cluster head indicators
        n_observation = (6 * (self.config.network.num_sensor+1) + 3)*2
        print(f"Observation space: {n_observation}")
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_observation,), dtype=np.float32)
        self.prev_obs = np.zeros(int(n_observation/2))
        self.action_space = spaces.Discrete(n_actions)

    def _get_obs(self):
        obs = hrl.add_rm_obs(num_sensors=self.config.network.num_sensor,
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

    def _create_network(self):

        for node in self.network.nodes.values():
            node.is_cluster_head = False
            node.cluster_id = 0
            # use np random to set the energy
            node.energy = np.random.uniform(
                low=0.5, high=self.config.network.protocol.init_energy)

        # Choose 5% of the number of nodes as cluster heads
        num_cluster_heads = int(
            self.config.network.num_sensor * self.config.network.protocol.cluster_head_percentage)

        # Choose num_cluster_heads nodes as cluster heads from the set of nodes
        # whose energy is greater or equal to the current network's average energy
        avg_energy = self.network.average_energy()
        # Also avoid choosing the sink as a cluster head
        cluster_heads = np.random.choice(
            [node for node in self.network.nodes.values() if node.energy >= avg_energy and node.node_id != 1], size=num_cluster_heads, replace=False)

        # Set the cluster heads
        for cluster_head in cluster_heads:
            cluster_head.is_cluster_head = True
            cluster_head.cluster_id = cluster_head.node_id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.acc_reward = 0
        self.round = 0

        # Create a random network of num_nodes nodes and random positions and
        # random energy levels
        self._create_network()

        # print all cluster heads
        # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        # ) if cluster_head.is_cluster_head]
        # print(f"Cluster heads: {chs}")

        leach.create_clusters(self.network)

        self._dissipate_energy()

        observation, info = self._get_obs()

        return observation, info

    def _dissipate_energy(self):
        hrl.dissipate_energy(round=self.round, network=self.network,
                             elect=self.elect, eda=self.eda,
                             packet_size=self.packet_size, eamp=self.eamp)

    def step(self, action):
        self.round += 1
        self.action = action
        action = self.actions_dict[action]
        node = self.network.nodes[action]
        done = False

        if node.energy < self.network.average_energy():
            # print(f"Node {node.node_id} is not a cluster head")
            obs, info = self._get_obs()
            return obs, 0, True, False, info

        if node.is_cluster_head:
            # remove the cluster head from the network
            node.is_cluster_head = False
            node.cluster_id = 0
            # print(f"Cluster head {node.node_id} removed")
        else:
            # add the cluster head to the network
            node.is_cluster_head = True
            node.cluster_id = node.node_id
            # print(f"Cluster head {node.node_id} added")

        leach.create_clusters(self.network)

        # Save the current energy
        current_energy = self.network.remaining_energy()

        self._dissipate_energy()

        # Latest energy
        latest_energy = self.network.remaining_energy()

        reward = 2-1*(current_energy-latest_energy)

        observation, info = self._get_obs()

        return observation, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
