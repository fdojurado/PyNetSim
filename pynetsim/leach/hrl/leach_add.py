import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pynetsim.leach as leach


class LEACH_ADD(gym.Env):
    def __init__(self, network):
        self.name = "SelectCH"
        self.config = network.config
        self.network = network

        # Energy conversion factors
        self.elect = self.config.network.protocol.eelect_nano * 1e-9
        self.etx = self.config.network.protocol.etx_nano * 1e-9
        self.erx = self.config.network.protocol.erx_nano * 1e-9
        self.eamp = self.config.network.protocol.eamp_pico * 1e-12
        self.eda = self.config.network.protocol.eda_nano * 1e-9

        self.packet_size = self.config.network.protocol.packet_size
        self.round = 0

        # Create the action space. It ranges from NODEID 2 to num_nodes
        # The sink is not a cluster head
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
        n_observation = 2 * (self.config.network.num_sensor+1)
        print(f"Observation space: {n_observation}")
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_observation,), dtype=np.float32)

        self.action_space = spaces.Discrete(n_actions)

    def _get_obs(self):
        # Put the energy consumption in a numpy array
        energy_consumption = np.zeros(self.config.network.num_sensor+1)
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            energy = max(node.energy, 0)/2
            energy_consumption[node.node_id] = energy

        # print(f"sizes: {len(energy_consumption)}, {len(self.network.nodes)}")

        cluster_heads = np.zeros(self.config.network.num_sensor+1)
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            if node.is_cluster_head:
                cluster_heads[node.node_id] = 1

        # print(f"Size of cluster heads: {len(cluster_heads)}")

        observation = np.append(energy_consumption, cluster_heads)
        # append rounds
        # observation = np.append(observation, self.round)
        info = {}

        return observation, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.acc_reward = 0
        self.round = 0

        cluster_head_id = self.np_random.choice(
            list(self.actions_dict.values()))
        print("Resetting environment with cluster head: ", cluster_head_id)
        cluster_head = self.network.get_node(cluster_head_id)

        # Set all dst_to_sink for all nodes
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            node.dst_to_sink = self.network.distance_to_sink(node)

        for node in self.network.nodes.values():
            node.is_cluster_head = False
            node.cluster_id = 0
            node.energy = self.config.network.protocol.init_energy

        cluster_head.is_cluster_head = True
        cluster_head.cluster_id = cluster_head.node_id

        # print all cluster heads
        chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        ) if cluster_head.is_cluster_head]
        print(f"Cluster heads: {chs}")

        leach.create_clusters(self.network)

        self.dissipate_energy()

        observation, info = self._get_obs()

        return observation, info

    def dissipate_energy(self):
        leach.energy_dissipation_non_cluster_heads(round=self.round, network=self.network,
                                                   elect=self.elect, eda=self.eda,
                                                   packet_size=self.packet_size, eamp=self.eamp)
        leach.energy_dissipation_cluster_heads(round=self.round, network=self.network,
                                               elect=self.elect, eda=self.eda,
                                               packet_size=self.packet_size, eamp=self.eamp)

    def step(self, action):
        self.round += 1
        action = self.actions_dict[action]
        cluster_head = self.network.get_node(action)
        done = False

        # print(
        #     f"Selected cluster head: {cluster_head.node_id} in round {self.round}")

        if cluster_head.energy <= 0:
            self.dissipate_energy()
            obs, info = self._get_obs()
            print(f"Cluster head {cluster_head.node_id} is dead.")
            return obs, 0, True, False, info

        if cluster_head.is_cluster_head:
            self.dissipate_energy()
            obs, info = self._get_obs()
            print(
                f"Node {cluster_head.node_id} is already a cluster head in round {self.round}.")
            return obs, 0, True, False, info

        cluster_head.is_cluster_head = True
        cluster_head.cluster_id = cluster_head.node_id

        leach.create_clusters(self.network)

        # Save current energy
        current_energy = self.network.remaining_energy()

        self.dissipate_energy()

        # latest energy
        latest_energy = self.network.remaining_energy()

        # The reward is the difference between the latest energy and the current energy
        # but we want to get better rewards for lower differences
        reward = 2-1*(current_energy - latest_energy)
        # reward = 1

        self.acc_reward += reward

        # print(
        #     f"Curr energy: {current_energy}, latest energy: {latest_energy}, diff: {current_energy - latest_energy}, reward: {reward}, acc_reward: {self.acc_reward}")
        # input("Press Enter to continue...")

        observation, info = self._get_obs()

        nodes_available = any(
            not node.is_cluster_head for node in self.network.nodes.values() if node.node_id != 1)
        if not nodes_available:
            print("No nodes available. Well done! Round: ", self.round)
            input("Press Enter to continue...")
            done = True

        # print(f"NODE ID {cluster_head.node_id} is the cluster head")

        # print all cluster heads
        # chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        # ) if cluster_head.is_cluster_head]
        # print(f"Cluster heads: {chs}")

        # print(f"observation: {observation}")
        # input("Press Enter to continue...")

        # plot
        # fig, ax = plt.subplots()
        # leach.plot_clusters(network=self.network, round=self.round, ax=ax)

        # plt.show()

        # input("Press Enter to continue...")

        return observation, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
