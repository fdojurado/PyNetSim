# This script runs a simulation multiple times with different seeds.

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pynetsim.common as common
import gymnasium as gym
import argparse
import sys
import os

from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS
from pynetsim.utils import PyNetSimLogger

from stable_baselines3.common.monitor import Monitor
from pynetsim.leach.rl.leach_rl_milp import LEACH_RL_MILP
from rich.progress import Progress
from stable_baselines3 import DQN


SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config.yml")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log", namespace="Main")
logger = logger_utility.get_logger()

def run_with_plotting(config, network, model, network_model, rounds):
    fig, ax = plt.subplots()
    common.plot_clusters(network=network, round=0, ax=ax)

    def animate(round, network=network):
        round = evaluate_round(round, config, network,
                               model, network_model, rounds)

        if round >= rounds or network.alive_nodes() <= 0:
            print("Done!")
            ani.event_source.stop()

        ax.clear()

        plt.pause(2)

    ani = animation.FuncAnimation(
        fig, animate, frames=range(1, rounds+1), repeat=False, fargs=(network, ))

    plt.show()


def run_without_plotting(config, network, model, network_model, rounds, name):
    round = 0
    with Progress() as progress:
        task = progress.add_task("[cyan]Simulation Progress", total=rounds)

        env = create_env(config, network, network_model)
        obs, _ = env.reset(options={"round": round})

        done = False

        while network.alive_nodes() > 0 and round < rounds and not done:
            print(f"Round: {round}")
            # Start the progress bar
            round, obs, info, done = evaluate_round(
                round, config, network, model, network_model, round, env, obs)
            print(f"Alive nodes: {network.alive_nodes()}")
            # Update the progress bar
            progress.update(task, completed=round)
        info_network = info["network"]
        info_net_model = info["network_model"]
        # export the metrics
        info_network.set_stats_name(name)
        info_network.export_stats()
        progress.update(task, completed=rounds)


def create_env(config, network, network_model):
    # Create the environment
    # env_name = config.network.protocol.name
    env = LEACH_RL_MILP(network, network_model, config, test=True)
    env = gym.wrappers.TimeLimit(
        env, max_episode_steps=config.network.protocol.max_steps)
    env = Monitor(env, args.log)
    return env


def print_energy_consumption_difference(network, network_copy):
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        print(
            f"Node {node.node_id}: {node.remaining_energy}, {network_copy.nodes[node.node_id].energy}")


def update_cluster_heads(network, network_copy):
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        node.is_cluster_head = network_copy.nodes[node.node_id].is_cluster_head
        node.cluster_id = network_copy.nodes[node.node_id].cluster_id
    chs = [cluster_head.node_id for cluster_head in network.nodes.values()
           if cluster_head.is_cluster_head]
    print(f"Cluster heads at high level: {chs}")


def evaluate_round(round, config, network, model, network_model, rounds, env, obs):
    rounds += 1
    action, _ = model.predict(obs)
    print(f"Action taken: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Reward: {reward}, done: {done}")

    return rounds, obs, info, done


def evaluate(config, network, model, network_model, rounds, plot, name):

    # Load the model
    model = DQN.load(model)
    if plot:
        run_with_plotting(config, network, model, network_model, rounds)
        return
    run_without_plotting(config, network, model, network_model, rounds, name)


def main(arguments):
    """
    Main function

    :param arguments: Arguments
    :type arguments: argparse.Namespace

    :return: None
    """
    # Load config
    config = load_config(arguments.config)
    logger.info("Loading config from %s", arguments.config)

    # generate random seeds
    seeds = [i for i in range(arguments.runs)]
    for seed in seeds:
        config.seed = seed
        network = Network(config=config)
        network_model = NETWORK_MODELS[config.network.model](
            config=config, network=network)
        network.set_model(network_model)
        network.initialize()
        name = f"{arguments.name}_{seed}"
        network_model.init()
        evaluate(config, network, arguments.model, network_model,
                 rounds=config.network.protocol.rounds,
                 plot=arguments.plot, name=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="Pre-trained model to load", default=None)
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    parser.add_argument("-l", "--log", help="Log directory", required=True)
    # how many times to run the simulation
    parser.add_argument("--runs", "-r", type=int,
                        help="Number of runs", default=5)
    # Name of the simulation
    parser.add_argument("--name", "-n", type=str,
                        help="Name of the simulation", default="leach_run")
    parser.add_argument("-p", "--plot", help="Plot the network",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
