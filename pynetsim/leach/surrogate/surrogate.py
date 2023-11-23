import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pyomo.environ as pyo
import pynetsim.leach.leach_milp as leach_milp


from pynetsim.utils import PyNetSimLogger
from rich.progress import Progress
from pynetsim.leach.surrogate.cluster_heads import ClusterHeadModel
from pynetsim.leach.surrogate.cluster_assignment import ClusterAssignmentModel

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log", namespace=__name__)
logger = logger_utility.get_logger()


class SurrogateModel:

    def __init__(self, config, network=object, net_model=object):
        self.name = "Surrogate Model"
        self.net_model = net_model
        self.config = config
        self.network = network
        self.cluster_head_model = ClusterHeadModel(config=config,
                                                   network=network)
        self.cluster_assignment_model = ClusterAssignmentModel(config=config,
                                                               network=network)

    def run(self):
        logger.info(
            f"Running {self.name}")
        # self.metrics = {}
        # self.model, _, _ = self.surrogate_model.get_model()
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

    def print_clusters(self):
        # Print cluster assignments
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            logger.info(
                f"Node {node.node_id} cluster head: {node.cluster_id}")

    def predict_cluster_heads(self, round):
        # Get the cluster heads
        cluster_heads = self.cluster_head_model.predict(
            network=self.network, round=round)
        logger.info(f"Predicted cluster heads: {cluster_heads}")
        return cluster_heads

    def predict_cluster_assignments(self, cluster_heads):
        # Get the cluster assignments
        cluster_assignments = self.cluster_assignment_model.predict(
            network=self.network, cluster_heads=cluster_heads)
        logger.info(f"Predicted cluster heads: {cluster_heads}")
        logger.info(f"Predicted cluster assignments: {cluster_assignments}")
        # Now match each element in cluster assignments whose value is the index of the cluster head
        # to the actual cluster head
        for i, cluster_head in enumerate(cluster_assignments):
            cluster_assignments[i] = cluster_heads[cluster_head]
        logger.info(f"Predicted cluster assignments: {cluster_assignments}")
        return cluster_assignments

    def set_cluster_heads(self, cluster_heads):
        for node in self.network:
            if node.node_id in cluster_heads:
                self.network.mark_as_cluster_head(node, node.node_id)
            else:
                self.network.mark_as_non_cluster_head(node)

    def set_clusters(self, cluster_assignments):
        for i, cluster_head in enumerate(cluster_assignments):
            node = self.network.get_node(i+2)
            node.cluster_id = int(cluster_head)
            node.dst_to_cluster_head = self.network.distance_between_nodes(
                node, self.network.get_node(cluster_head))

    def evaluate_round(self, round):
        round += 1
        print(f"Round {round}")

        self.max_chs = int(self.network.alive_nodes() *
                           self.config.network.protocol.cluster_head_percentage) + 1
        # Create cluster assignments predicted by the surrogate model
        cluster_heads = self.predict_cluster_heads(round=round)
        # Set cluster heads
        self.set_cluster_heads(cluster_heads)
        # Lets predict the cluster assignments
        cluster_assignments = self.predict_cluster_assignments(
            cluster_heads=cluster_heads)
        self.set_clusters(cluster_assignments)
        self.net_model.dissipate_energy(round=round)
        input("Press enter to continue...")
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
