#     PyNetSim: A Python-based Network Simulator for Low-Energy Adaptive Clustering Hierarchy (LEACH) Protocol
#     Copyright (C) 2024  F. Fernando Jurado-Lasso (ffjla@dtu.dk)

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import sys
import os

from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS
from pynetsim.utils import PyNetSimLogger


SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config.yml")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log", namespace="Main")
logger = logger_utility.get_logger()


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
        network.stats.name = f"{arguments.name}_{seed}"
        network_model.init()
        network.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    # how many times to run the simulation
    parser.add_argument("--runs", "-r", type=int,
                        help="Number of runs", default=5)
    # Name of the simulation
    parser.add_argument("--name", "-n", type=str,
                        help="Name of the simulation", default="leach_run")
    args = parser.parse_args()
    main(args)
    sys.exit(0)
