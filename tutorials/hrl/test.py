# This tutorial simple constructs a 200x200 network with 20 nodes and a transmission range of 80.
# The network is plotted using matplotlib.
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pynetsim.leach.hrl as hrl
import gymnasium as gym
import numpy as np
import argparse
import copy
import sys
import os

from stable_baselines3.common.monitor import Monitor
from pynetsim.network.network import Network
from pynetsim.config import PyNetSimConfig
from stable_baselines3 import DQN
from pynetsim.config import PROTOCOLS


SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config_leach_rm.json")


def run_with_plotting(config, network, model, rounds,
                      network_energy, num_dead_nodes, num_alive_nodes):
    fig, ax = plt.subplots()
    hrl.plot_clusters(network=network, round=0, ax=ax)

    def animate(round, network=network):
        round += 1
        print(f"Round: {round}")
        done = False
        network_copy = copy.deepcopy(network)
        env = create_env(config, network_copy)
        obs, _ = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break

        # Compare the energy consumption of the original network and the copy
        for node in network.nodes.values():
            if node.node_id == 1:
                continue
            # print original and copy energy
            print(
                f"Node {node.node_id}: {node.energy}, {network_copy.nodes[node.node_id].energy}")

        # Copy the cluster heads selection into the original network
        for node in network.nodes.values():
            if node.node_id == 1:
                continue
            node.is_cluster_head = network_copy.nodes[node.node_id].is_cluster_head
            node.cluster_id = network_copy.nodes[node.node_id].cluster_id

        # print all cluster heads
        chs = [cluster_head.node_id for cluster_head in network.nodes.values(
        ) if cluster_head.is_cluster_head]
        print(f"Cluster heads: {chs}")

        hrl.create_clusters(network)
        hrl.dissipate_energy(round=round, network=network,
                             elect=config.network.protocol.eelect_nano*1e-9,
                             eda=config.network.protocol.eda_nano*1e-9,
                             packet_size=config.network.protocol.packet_size,
                             eamp=config.network.protocol.eamp_pico*1e-12)

        # print energy levels
        for node in network.nodes.values():
            if node.node_id == 1:
                continue
            print(f"Node {node.node_id}: {node.energy}")

        if round >= rounds or network.alive_nodes() <= 0:
            print("Done!")
            ani.event_source.stop()

        ax.clear()

        hrl.plot_clusters(network=network, round=round, ax=ax)
        hrl.store_metrics(config, network, round, network_energy,
                          num_dead_nodes, num_alive_nodes)

        hrl.save_metrics(config, "LEACH-RL", network_energy,
                         num_dead_nodes, num_alive_nodes)

        plt.pause(2)

    ani = animation.FuncAnimation(
        fig, animate, frames=range(1, rounds+1), repeat=False, fargs=(network, ))

    plt.show()


def run_without_plotting(config, network, model, rounds,
                         network_energy, num_dead_nodes, num_alive_nodes):
    round = 0
    while network.alive_nodes() > 0 and round < rounds:
        round += 1
        print(f"Round: {round}")
        done = False
        network_copy = copy.deepcopy(network)
        env = create_env(config, network_copy)
        obs, _ = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break

        # Compare the energy consumption of the original network and the copy
        for node in network.nodes.values():
            if node.node_id == 1:
                continue
            # print original and copy energy
            print(
                f"Node {node.node_id}: {node.energy}, {network_copy.nodes[node.node_id].energy}")

        # Copy the cluster heads selection into the original network
        for node in network.nodes.values():
            if node.node_id == 1:
                continue
            node.is_cluster_head = network_copy.nodes[node.node_id].is_cluster_head
            node.cluster_id = network_copy.nodes[node.node_id].cluster_id

        # print all cluster heads
        chs = [cluster_head.node_id for cluster_head in network.nodes.values(
        ) if cluster_head.is_cluster_head]
        print(f"Cluster heads: {chs}")

        hrl.create_clusters(network)
        hrl.dissipate_energy(round=round, network=network,
                             elect=config.network.protocol.eelect_nano*1e-9,
                             eda=config.network.protocol.eda_nano*1e-9,
                             packet_size=config.network.protocol.packet_size,
                             eamp=config.network.protocol.eamp_pico*1e-12)

        # print energy levels
        for node in network.nodes.values():
            if node.node_id == 1:
                continue
            print(f"Node {node.node_id}: {node.energy}")

        hrl.store_metrics(config, network, round, network_energy,
                          num_dead_nodes, num_alive_nodes)

        hrl.save_metrics(config, "LEACH-RL", network_energy,
                         num_dead_nodes, num_alive_nodes)


def create_env(config, network):
    # Create the environment
    env_name = config.network.protocol.name
    env = PROTOCOLS[env_name](network)
    env = gym.wrappers.TimeLimit(
        env, max_episode_steps=config.network.protocol.max_steps)
    env = Monitor(env, args.log)
    return env


def evaluate(config, network, model, rounds, plot):

    network_energy = {}
    num_dead_nodes = {}
    num_alive_nodes = {}
    # Load the model
    model = DQN.load(model)
    if plot:
        run_with_plotting(config, network, model, rounds,
                          network_energy, num_dead_nodes, num_alive_nodes)
        return
    run_without_plotting(config, network, model, rounds,
                         network_energy, num_dead_nodes, num_alive_nodes)


def main(args):
    # If there is not configuration file, use the default one
    if args.config is None:
        config = PyNetSimConfig.from_json(CONFIG_FILE)
    else:
        config = PyNetSimConfig.from_json(args.config)
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    # Create the network
    network = Network(config)
    network.initialize()

    evaluate(config, network, args.model,
             rounds=config.network.protocol.rounds,
             plot=args.plot)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # tensologger file location
    argparser.add_argument(
        "-m", "--model", help="Model to load", required=True)
    # Path to the configuration file (.json)
    argparser.add_argument("-c", "--config", help="Configuration file")
    # Path to the log directory, required.
    argparser.add_argument("-l", "--log", help="Log directory", required=True)
    # A boolean flag to indicate whether to plot the network or not
    argparser.add_argument("-p", "--plot", help="Plot the network",
                           action="store_true", default=False)
    args = argparser.parse_args()
    main(args)
    sys.exit(0)
