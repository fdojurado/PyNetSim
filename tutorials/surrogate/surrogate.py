from pynetsim.config import load_config
from pynetsim.utils import PyNetSimLogger
from pynetsim.network.network import Network
from pynetsim.config import NETWORK_MODELS
from pynetsim.leach.surrogate.surrogate import SurrogateModel
from rich.logging import RichHandler

import sys
import os
import argparse
import logging.config


SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "surrogate.yml")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log")
logger = logger_utility.get_logger()


def main(args):
    # Load config
    config = load_config(args.config)
    logger.info(f"Loading config from {args.config}")

    network = Network(config=config)
    network_model = NETWORK_MODELS[config.network.model](
        config=config, network=network)
    network.set_model(network_model)
    network.initialize()
    # initialize the network model
    network_model.init()

    # Instantiate the model
    surrogate_model = SurrogateModel(config=config, network=network,
                                     net_model=network_model)

    # Run
    surrogate_model.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
