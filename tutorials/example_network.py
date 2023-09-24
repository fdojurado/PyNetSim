# This tutorial simple constructs a 200x200 network with 20 nodes and a transmission range of 80.
# The network is plotted using matplotlib.

from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS

import sys
import os

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config.yml")


def main():
    # Load config
    config = load_config(CONFIG_FILE)
    print(f"config: {config}")

    network = Network(config=config)
    network_model = NETWORK_MODELS[config.network.model](
        config=config, network=network)
    network.set_model(network_model)
    network.initialize()
    network.start()


if __name__ == "__main__":
    main()
    sys.exit(0)
