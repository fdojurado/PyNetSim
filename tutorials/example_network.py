# This tutorial simple constructs a 200x200 network with 20 nodes and a transmission range of 80.
# The network is plotted using matplotlib.

import argparse
import sys
import os

from pynetsim import PyNetSim
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
    # Create a PyNetSim instance
    pynetsim = PyNetSim(config=arguments.config, print_rich=True)

    pynetsim.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
