# This script collects data for the MILP model.
# The main idea is to run the LEACH-CE-E protocol with different weights.
import pynetsim.leach.leach_milp as leach_milp
import sys
import os

from pynetsim.leach.leach_milp.leach_ce_e import LEACH_CE_E
from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS

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

    # Here, we want to collect data for the MILP model.
    # We are varying the weights for alpha, beta, gamma, a and b.

    # Lets first generate the weights which range from 0 to 5 each.
    # Lets use a 0.2 step size.

    # Lets generate the all the combinations of weights.

    max_alpha = 61
    max_beta = 61
    max_gamma = 61

    min_alpha = 1
    min_beta = 1
    min_gamma = 1

    step_size = 8

    for alpha in range(min_alpha, max_alpha, step_size):
        alpha_weight = alpha / 10
        for beta in range(min_beta, max_beta, step_size):
            beta_weight = beta / 10
            for gamma in range(min_gamma, max_gamma, step_size):
                gamma_weight = gamma / 10
                network_copy, network_model_copy = leach_milp.copy_network(
                    network, network_model)
                # Lets create the object of the LEACH-CE-E protocol.
                leach_ce_e = LEACH_CE_E(network_copy, network_model_copy,
                                        alpha=alpha_weight, beta=beta_weight, gamma=gamma_weight)
                network_copy.stats.name = f"LEACH-CE-E_{alpha_weight}_{beta_weight}_{gamma_weight}"
                # Lets run the protocol.
                leach_ce_e.run()


if __name__ == "__main__":
    main()
    sys.exit(0)
