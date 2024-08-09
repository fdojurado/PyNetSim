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


from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS
from pynetsim.utils import PyNetSimLogger
from stable_baselines3 import DQN
from pynetsim.leach.rl.leach_rl import LEACH_RL
from pynetsim.leach.rl.leach_rl_milp import LEACH_RL_MILP
import gymnasium as gym

import numpy as np
import argparse
import sys
import os

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config.yml")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log", namespace="Main")
logger = logger_utility.get_logger()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    logger.info(f"Num timesteps: {self.num_timesteps}")
                    logger.info(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        logger.info(
                            f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


def main(args):
    # Check if the protocol is valid
    if args.protocol not in ["LEACH_RL", "LEACH_RL_MILP"]:
        logger.error(
            f"Protocol {args.protocol} is not valid. Please choose between LEACH_RL and LEACH_RL_MILP")
        sys.exit(1)
    # Create tensorboard logger folder if it does not exist
    if not os.path.exists(args.tensorboard):
        os.makedirs(args.tensorboard)
    # Create log dir folder if it does not exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # Load config
    config = load_config(CONFIG_FILE)
    logger.info(f"Loading config from {CONFIG_FILE}")

    network = Network(config=config)
    network_model = NETWORK_MODELS[config.network.model](
        config=config, network=network)
    network.set_model(network_model)
    network.initialize()
    network_model.init()
    # -----------Lets train the DQN network
    if args.protocol == "LEACH_RL_MILP":
        env = LEACH_RL_MILP(network, network_model, config)
    else:
        env = LEACH_RL(network, network_model, config)
    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=config.network.protocol.max_steps
    )
    env = Monitor(env, args.save)
    tensorboard_log = args.tensorboard
    best_model = SaveOnBestTrainingRewardCallback(
        check_freq=100, log_dir=args.save)
    if args.model is not None:
        model = DQN.load(args.model, env=env, tensorboard_log=tensorboard_log)
        logger.info(f"Loaded model from {args.model}")
    else:
        logger.info("Training new model")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            # buffer_size=50000,
            learning_starts=512,
            batch_size=128,
            # tau=1.0,
            gamma=0.90,
            # train_freq=4,
            target_update_interval=100,
            exploration_fraction=0.8,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log=tensorboard_log
        )
    model.learn(total_timesteps=200e3, log_interval=4, callback=best_model)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--model", type=str,
                           help="Pre-trained model to load", default=None)
    # Select between LEACH_RL and LEACH_RL_MILP
    argparser.add_argument("-p", "--protocol", type=str,
                           help="Choose between LEACH_RL and LEACH_RL_MILP", default="LEACH_RL")
    argparser.add_argument("-t", "--tensorboard", type=str,
                           default="./dqn_leach_add_tensorboard/")
    # path to save the model
    argparser.add_argument("-s", "--save", type=str,
                           default="./log/")
    args = argparser.parse_args()
    logger.info(f"Using protocol {args.protocol}")
    main(args)
    sys.exit(0)
