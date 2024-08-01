# This tutorial simple constructs a 200x200 network with 20 nodes and a transmission range of 80.
# The network is plotted using matplotlib.

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

    network = Network(config=config)
    network_model = NETWORK_MODELS[config.network.model](
        config=config, network=network)
    network.set_model(network_model)
    network.initialize()
    network_model.init()
    network.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
