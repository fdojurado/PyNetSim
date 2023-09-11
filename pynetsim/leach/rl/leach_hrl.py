import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import pynetsim.leach.rl as rl
import types
import os
import copy
import gymnasium as gym

# from pynetsim.config import PyNetSimConfig
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from pynetsim.leach.rl.leach_rl import LEACH_RL
from gymnasium import spaces

action_types = types.SimpleNamespace()
action_types.NEW_CLUSTER = 0
action_types.SAME_CLUSTER = 1

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
PRE_TRAINED_MODELS_PATH = os.path.normpath(
    os.path.join(SELF_PATH, "pre_trained"))
PRE_TRAINED_MODEL = os.path.join(PRE_TRAINED_MODELS_PATH, "rl.zip")


class LEACH_HRL(gym.Env):
    def __init__(self, network, net_model: object):
        self.name = "LEACH-HRL"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.max_steps = self.config.network.protocol.max_steps
        self.num_sensor = self.config.network.num_sensor
        self.action = 0

        self.round = 0

        # Calculate the maximum distance to the sink, which is the distance
        # between the sink and the farthest node
        self.max_distance = self.network.max_distance_to_sink()

        self.dst_to_sink = np.zeros(self.num_sensor+1)
        # Set all dst_to_sink for all nodes
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            node.dst_to_sink = self.network.distance_to_sink(node)
            self.dst_to_sink[node.node_id] = node.dst_to_sink/self.max_distance

        # Initialize array of nodes' positions
        self.x_locations = np.zeros(self.num_sensor+1)
        self.y_locations = np.zeros(self.num_sensor+1)
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            self.x_locations[node.node_id] = node.x/self.config.network.width
            self.y_locations[node.node_id] = node.y/self.config.network.height

        # Add pre-trained lower level agent
        self.network_copy = copy.deepcopy(self.network)
        self.net_model_copy = copy.deepcopy(self.net_model)
        self.lower_level_env = LEACH_RL(network=self.network_copy,
                                        net_model=self.net_model_copy)
        self.lower_level_env = gym.wrappers.TimeLimit(
            self.lower_level_env, max_episode_steps=10)
        self.lower_level_env = Monitor(self.lower_level_env)
        self.lower_level_agent = DQN.load(PRE_TRAINED_MODEL)
        self.lower_level_agent.set_env(self.lower_level_env)
        self.lower_level_env.reset()

        self.n_actions = len(action_types.__dict__)
        print(f"Action space: {self.n_actions}")

        # Observation space: energy consumption + cluster head indicators
        n_observation = (6 * (self.num_sensor+1) + 3)*2
        print(f"Observation space: {n_observation}")
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_observation,), dtype=np.float32)
        self.prev_obs = np.zeros(int(n_observation/2))
        self.action_space = spaces.Discrete(self.n_actions)

    def _get_obs(self):
        obs = rl.obs(num_sensors=self.num_sensor,
                     network=self.network,
                     x_pos=self.x_locations,
                     y_pos=self.y_locations,
                     dst_to_sink=self.dst_to_sink,
                     init_energy=self.config.network.protocol.init_energy,
                     round=self.round,
                     max_steps=self.max_steps,
                     max_distance=self.max_distance,
                     action_taken=self.action/self.n_actions)

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
        reward = 1
        return reward

    def _update_lower_level_env(self):
        self.lower_level_env.set_network(self.network_copy)
        self.lower_level_env.set_net_model(self.net_model_copy)

    def _update_cluster_heads(self):
        for node in self.network.nodes.values():
            if node.node_id == 1:
                continue
            node.is_cluster_head = self.network_copy.nodes[node.node_id].is_cluster_head
            node.cluster_id = self.network_copy.nodes[node.node_id].cluster_id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.round = 0

        # Create a random network of num_nodes nodes and random positions and
        # random energy levels
        rl.create_network(network=self.network, config=self.config,
                          lower_energy=self.config.network.protocol.init_energy/2)

        # print all cluster heads
        chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        ) if cluster_head.is_cluster_head]
        print(f"Cluster heads at reset: {chs}")

        self.network.create_clusters()

        self.net_model.dissipate_energy(round=self.round)

        observation, info = self._get_obs()

        return observation, info

    def step(self, action_type):
        self.round += 1
        done = False
        self.action = int(action_type)
        print(f"Action taken: {action_type}")
        lower_level_env_done = False
        if action_type == action_types.NEW_CLUSTER:
            self.network_copy = copy.deepcopy(self.network)
            self.network_model_copy = copy.deepcopy(self.net_model)
            self._update_lower_level_env()
            obs, _ = self._get_obs()
            while not lower_level_env_done:
                action, _ = self.lower_level_agent.predict(
                    obs, deterministic=True)
                print(f"Lower level action: {action}")
                obs, reward, terminated, truncated, _ = self.lower_level_env.step(
                    action)
                lower_level_env_done = terminated or truncated
                if lower_level_env_done:
                    print(f"Lower level env done: {lower_level_env_done}")
                    self.lower_level_env.reset()
                    break
            # Update the cluster heads
            self._update_cluster_heads()
            self.network.create_clusters()
        elif action_type == action_types.SAME_CLUSTER:
            pass

        chs = [cluster_head.node_id for cluster_head in self.network.nodes.values(
        ) if cluster_head.is_cluster_head]
        print(f"Cluster heads: {chs}")

        reward = self._calculate_reward()
        observation, info = self._get_obs()
        # we are done when the first node dies
        for node in self.network:
            if node.node_id == 1:
                continue
            if node.remaining_energy <= 0:
                # print(f"LEACH-HRL: Node {node.node_id} died")
                done = True
                break

        input("Press Enter to continue...")

        return observation, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
