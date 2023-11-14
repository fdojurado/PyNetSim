from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS
from pynetsim.utils import PyNetSimLogger
from pynetsim.leach.surrogate.ch_regression import SurrogateModel
from pynetsim.leach.surrogate.cluster_assignment import ClusterAssignment

import sys
import os
import argparse

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "surrogate.yml")

MODELS = {
    "cluster_head": SurrogateModel,
    "cluster_assignment": ClusterAssignment
}

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log")
logger = logger_utility.get_logger()


def main(args):
    # Load config
    config = load_config(args.config)
    logger.info(f"Loading config from {args.config}")

    # Select the model to train
    model_name = args.model_name
    if model_name is None:
        raise Exception("Please provide the name of the model to train")
    if model_name not in MODELS:
        raise Exception(
            f"Model name should be one of {list(MODELS.keys())}")

    # Instantiate the model
    surrogate_model = MODELS[model_name](config=config)

    # Initialize the surrogate model
    surrogate_model.init()
    # Train the surrogate model
    surrogate_model.train()
    # surrogate_model.test(print_output=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    parser.add_argument("--model_name", "-m", type=str,
                        help="Name of the model to train", default=None)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
