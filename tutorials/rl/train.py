# This tutorial simple constructs a 200x200 network with 20 nodes and a transmission range of 80.
# The network is plotted using matplotlib.

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from pynetsim.network.network import Network
from pynetsim.config import PyNetSimConfig
from stable_baselines3 import DQN, PPO
from pynetsim.config import PROTOCOLS
import gymnasium as gym

import numpy as np
import argparse
import sys
import os

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config_leach_rm.json")


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
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


def main(args):
    # Create tensorboard logger folder if it does not exist
    if not os.path.exists(args.tensorboard):
        os.makedirs(args.tensorboard)
    # Create log dir folder if it does not exist
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # Load config
    config = PyNetSimConfig.from_json(CONFIG_FILE)
    print(f"config: {config}")

    network = Network(config=config)
    network.initialize()
    # -----------Lets train the DQN network
    env_name = config.network.protocol.name
    env = PROTOCOLS[env_name](network)
    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=config.network.protocol.max_steps
    )
    env = Monitor(env, args.logdir)
    tensorboard_log = args.tensorboard
    best_model = SaveOnBestTrainingRewardCallback(
        check_freq=100, log_dir=args.logdir)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        # buffer_size=50000,
        learning_starts=5e3,
        batch_size=512,
        # tau=1.0,
        gamma=0.8,
        # train_freq=4,
        target_update_interval=100,
        exploration_fraction=0.8,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log=tensorboard_log
    )
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     n_steps=512,
    #     learning_rate=3e-4,
    #     batch_size=64,
    #     ent_coef=0.1,
    #     gamma=0.99,
    #     # buffer_size=50000,
    #     # learning_starts=5e3,
    #     # batch_size=512,
    #     # # tau=1.0,
    #     # gamma=0.8,
    #     # # train_freq=4,
    #     # target_update_interval=100,
    #     # exploration_fraction=0.8,
    #     # exploration_initial_eps=1.0,
    #     # exploration_final_eps=0.05,
    #     tensorboard_log=tensorboard_log
    # )
    model.learn(total_timesteps=500e3, log_interval=4, callback=best_model)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # tensologger file location
    argparser.add_argument("-t", "--tensorboard", type=str,
                           default="./dqn_leach_add_tensorboard/")
    # log dir
    argparser.add_argument("-l", "--logdir", type=str,
                           default="./dqn_leach_add_log/")
    args = argparser.parse_args()
    main(args)
    sys.exit(0)
