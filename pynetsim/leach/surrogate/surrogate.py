import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from pynetsim.utils import PyNetSimLogger
from rich.progress import Progress
from pynetsim.leach.surrogate.model import SurrogateModel, NetworkDataset
from torch.utils.data import DataLoader

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log", namespace=__name__)
logger = logger_utility.get_logger()


class SURROGATE:

    def __init__(self, network, net_model: object):
        self.name = "SURROGATE"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.largest_weight = self.config.surrogate.largest_weight
        self.alpha = 0.5
        self.beta = 2
        self.gamma = 1.7
        self.surrogate_model = SurrogateModel(config=self.config)

    def run(self):
        logger.info(f"Running {self.name}...")
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = False

        for node in self.network:
            node.is_cluster_head = False

        # Set all dst_to_sink for all nodes
        for node in self.network:
            node.dst_to_sink = self.network.distance_to_sink(node)

        if not plot_clusters_flag:
            self.run_without_plotting(
                num_rounds)
        else:
            self.run_with_plotting(
                num_rounds)

    def create_input_data(self):
        # We need to get alpha, beta, gamma, and energy levels for all nodes
        # We need to get alpha, beta, gamma for all nodes
        weights = [self.alpha/self.largest_weight, self.beta /
                   self.largest_weight, self.gamma/self.largest_weight]
        w_np = np.array(weights)
        logger.info(f"Weights: {w_np}, shape: {w_np.shape}")
        # print all energy levels
        # for node in self.network:
        #     logger.info(f"Node {node.node_id} energy level: {node.remaining_energy}")
        # We need to get energy levels for all nodes
        energy_levels = np.array(
            [node.remaining_energy for node in self.network if node.node_id != 1])
        logger.info(
            f"Energy levels: {energy_levels}, shape: {energy_levels.shape}")

        # We need to get the current membership of all nodes, this involves in getting the
        # cluster id of all nodes.
        cluster_ids = np.array(
            [0 if node.cluster_id is None else node.cluster_id for node in self.network])

        logger.info(f"Cluster ids: {cluster_ids}, shape: {cluster_ids.shape}")
        # Add a zero to the start of the cluster ids
        cluster_ids = np.insert(cluster_ids, 0, 0)

        logger.info(f"Cluster ids: {cluster_ids}, shape: {cluster_ids.shape}")

        # Now we concatenate weights, and energy levels
        input_data = np.concatenate((w_np, energy_levels))
        logger.info(f"Input data: {input_data}, shape: {input_data.shape}")

        return input_data, cluster_ids

    def predict_cluster_assignments(self):
        # Before we can predict the cluster assignments, we need to create the input data
        # for the surrogate model.
        numerical_data, categorical_data = self.create_input_data()
        logger.info(f"Shapes of numerical and categorical data: {numerical_data.shape}, {categorical_data.shape}")
        # Convert the numerical data to a tensor
        numerical_data =torch.from_numpy(numerical_data.astype(np.float32))
        # Convert the categorical data to a tensor
        categorical_data =torch.from_numpy(categorical_data.astype(np.int64))
        # Unsqueeze the numerical data
        numerical_data = numerical_data.unsqueeze(0)
        # Unsqueeze the categorical data
        categorical_data = categorical_data.unsqueeze(0)
        # print shapes
        logger.info(f"Shapes of numerical and categorical data: {numerical_data.shape}, {categorical_data.shape}")
        # Now we can predict the cluster assignments
        model, _, _ = self.surrogate_model.get_model()
        model.eval()
        logger.info(
            f"Shapes: {numerical_data.shape}, {categorical_data.shape}")
        with torch.no_grad():
                logger.info(
                    f"Categorical data: {categorical_data}, shape: {categorical_data.shape}")
                output = model(categorical_data=categorical_data,
                               numerical_data=numerical_data)
                logger.info(f"Output: {output}, shape: {output.shape}")
                _, predicted_assignments = torch.max(output, dim=1)
                logger.info(
                    f"Predicted cluster assignments: {predicted_assignments}")
        input("Press enter to continue...")

    def evaluate_round(self, round):
        round += 1

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        # Create cluster assignments predicted by the surrogate model
        cluster_assignments = self.predict_cluster_assignments()

        return round

    def run_without_plotting(self, num_rounds):
        round = 0
        with Progress() as progress:
            task = progress.add_task(
                f"[red]Running {self.name}...", total=num_rounds)
            while self.network.alive_nodes() > 0 and round < num_rounds:
                round = self.evaluate_round(round)
                progress.update(task, completed=round)
            progress.update(task, completed=num_rounds)

    def run_with_plotting(self, num_rounds):
        fig, ax = plt.subplots()
        common.plot_clusters(network=self.network, round=0, ax=ax)

        def animate(round):
            round = self.evaluate_round(round)

            if round >= num_rounds or self.network.alive_nodes() <= 0:
                logger.info("Done!")
                ani.event_source.stop()

            ax.clear()
            common.plot_clusters(network=self.network, round=round, ax=ax)

            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
