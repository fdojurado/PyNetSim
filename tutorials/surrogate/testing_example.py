from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS
from pynetsim.utils import PyNetSimLogger
from pynetsim.leach.surrogate.surrogate import SurrogateModel

import sys
import os
import argparse

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "surrogate_test.yml")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log")
logger = logger_utility.get_logger()


def main(args):
    # Load config
    config = load_config(args.config)
    logger.info(f"Loading config from {args.config}")

    # Create the surrogate model
    surrogate_model = SurrogateModel(config=config)

    # Initialize the surrogate model
    surrogate_model.init()

    # Test the surrogate model
    surrogate_model.test(
        batch=args.batch, print_output=args.print, weights=args.weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    parser.add_argument("-b", "--batch", type=int,
                        help="Batch size", default=64)
    parser.add_argument(
        "-p", "--print", help="Print the output", action="store_true"
    )
    # List of weights
    parser.add_argument("-w", "--weights", nargs="+", type=float,
                        help="Weights", default=None)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
