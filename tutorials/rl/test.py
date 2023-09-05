# This tutorial simple constructs a 200x200 network with 20 nodes and a transmission range of 80.
# The network is plotted using matplotlib.
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pynetsim.leach.rl as rl
import pynetsim.common as common
import gymnasium as gym
import numpy as np
import argparse
import copy
import sys
import os

from stable_baselines3.common.monitor import Monitor
from pynetsim.network.network import Network
from pynetsim.config import PyNetSimConfig, NETWORK_MODELS
from pynetsim.config import PROTOCOLS
from rich.progress import Progress
from stable_baselines3 import DQN


SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config.json")


def run_with_plotting(config, network, model, network_model, rounds,
                      network_energy, num_dead_nodes, num_alive_nodes,
                      num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                      control_packets_energy, control_packet_bits):
    fig, ax = plt.subplots()
    common.plot_clusters(network=network, round=0, ax=ax)

    def animate(round, network=network):
        round = evaluate_round(round, config, network,
                               model, network_model, rounds)

        if round >= rounds or network.alive_nodes() <= 0:
            print("Done!")
            ani.event_source.stop()

        ax.clear()

        common.plot_clusters(network=network, round=round, ax=ax)
        common.add_to_metrics(config, network, round, network_energy,
                              num_dead_nodes, num_alive_nodes, num_cluster_heads,
                              pkt_delivery_ratio, pkt_loss_ratio,
                              control_packets_energy, control_packet_bits)

        common.save_metrics(config, network_energy,
                            num_dead_nodes, num_alive_nodes, num_cluster_heads,
                            pkt_delivery_ratio, pkt_loss_ratio,
                            control_packets_energy, control_packet_bits)

        plt.pause(2)

    ani = animation.FuncAnimation(
        fig, animate, frames=range(1, rounds+1), repeat=False, fargs=(network, ))

    plt.show()


def run_without_plotting(config, network, model, network_model, rounds,
                         network_energy, num_dead_nodes, num_alive_nodes,
                         num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                         control_packets_energy, control_packet_bits):
    round = 0
    with Progress() as progress:
        task = progress.add_task("[cyan]Simulation Progress", total=rounds)

        while network.alive_nodes() > 0 and round < rounds:
            # Start the progress bar
            round = evaluate_round(
                round, config, network, model, network_model, rounds)
            common.add_to_metrics(config, network, round, network_energy,
                                  num_dead_nodes, num_alive_nodes,
                                  num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                                  control_packets_energy, control_packet_bits)
            common.save_metrics(config, network_energy,
                                num_dead_nodes, num_alive_nodes,
                                num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                                control_packets_energy, control_packet_bits)
            # Update the progress bar
            progress.update(task, completed=round)
        progress.update(task, completed=rounds)


def create_env(config, network, network_model):
    # Create the environment
    env_name = config.network.protocol.name
    env = PROTOCOLS[env_name](network, network_model)
    env = gym.wrappers.TimeLimit(
        env, max_episode_steps=config.network.protocol.max_steps)
    env = Monitor(env, args.log)
    return env


def print_energy_consumption_difference(network, network_copy):
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        print(
            f"Node {node.node_id}: {node.energy}, {network_copy.nodes[node.node_id].energy}")


def update_cluster_heads(network, network_copy):
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        node.is_cluster_head = network_copy.nodes[node.node_id].is_cluster_head
        node.cluster_id = network_copy.nodes[node.node_id].cluster_id
    chs = [cluster_head.node_id for cluster_head in network.nodes.values()
           if cluster_head.is_cluster_head]
    # print(f"Cluster heads: {chs}")


def evaluate_round(round, config, network, model, network_model, rounds):
    round += 1
    # print(f"Round: {round}")
    done = False
    network_copy = copy.deepcopy(network)
    network_model_copy = copy.deepcopy(network_model)
    env = create_env(config, network_copy, network_model_copy)
    obs, _ = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            break

    # print_energy_consumption_difference(network, network_copy)
    update_cluster_heads(network, network_copy)

    network.create_clusters()
    network_model.dissipate_energy(round=round)

    return round


def evaluate(config, network, model, network_model, rounds, plot):

    network_energy = {}
    num_dead_nodes = {}
    num_alive_nodes = {}
    num_cluster_heads = {}
    pkt_delivery_ratio = {}
    pkt_loss_ratio = {}
    control_packets_energy = {}
    control_packet_bits = {}
    # Load the model
    model = DQN.load(model)
    if plot:
        run_with_plotting(config, network, model, network_model, rounds,
                          network_energy, num_dead_nodes, num_alive_nodes,
                          num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                          control_packets_energy, control_packet_bits)
        return
    run_without_plotting(config, network, model, network_model, rounds,
                         network_energy, num_dead_nodes, num_alive_nodes,
                         num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                         control_packets_energy, control_packet_bits)


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
    network_model = NETWORK_MODELS[config.network.model](
        config=config, network=network)
    network.set_model(network_model)
    network.initialize()

    evaluate(config, network, args.model, network_model,
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
