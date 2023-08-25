# This tutorial simple constructs a 200x200 network with 20 nodes and a transmission range of 80.
# The network is plotted using matplotlib.

from pynetsim.network.network import Network
from pynetsim.config import PyNetSimConfig

import sys
import os

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config.json")



def main():
    # Load config
    config = PyNetSimConfig.from_json(CONFIG_FILE)
    print(f"config: {config}")

    network = Network(config=config)
    network.initialize()
    network.start()


if __name__ == "__main__":
    main()
    sys.exit(0)
