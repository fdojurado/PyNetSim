# This script aims to find the optimal weights (alpha, beta, gamma) for the MILP problem.
from pynetsim.utils import PyNetSimLogger
from pynetsim.config import load_config
from pynetsim.leach.surrogate.regression_model import SurrogateModel

import argparse
import sys
import os

INITIAL_WEIGHTS = [1, 1, 1]
TESTED_ROUND = 129

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "surrogate.yml")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log")
logger = logger_utility.get_logger()


def run_model(surrogate_model, weights):
    model, _, _ = surrogate_model.get_model(load_model=True)

    # Convert weights to tuple
    weights = tuple(weights)


def objective_function(surrogate_model, weights):
    # Lets predict the number of alive nodes up to the TESTED_ROUND
    run_model(surrogate_model, weights)


def simulated_annealing(surrogate_model):

    current_weights = INITIAL_WEIGHTS
    current_cost = objective_function(surrogate_model, current_weights)


def main(args):
    # Load config
    config = load_config(args.config)
    logger.info(f"Loading config from {args.config}")

    # Create the surrogate model
    surrogate_model = SurrogateModel(config=config)

    # Initialize the surrogate model
    surrogate_model.init()

    simulated_annealing(surrogate_model=surrogate_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
