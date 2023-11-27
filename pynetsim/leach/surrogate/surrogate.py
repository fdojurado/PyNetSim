import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pynetsim.leach.surrogate as leach_surrogate


from pynetsim.utils import PyNetSimLogger, Timer
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
            with Timer() as t:
                self.run_without_plotting(
                    num_rounds)
        else:
            self.run_with_plotting(
                num_rounds)

        # export the metrics
        self.network.export_stats()

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
            network=self.network, round=round, std_re=self.std_re, std_el=self.std_el,
            re=self.re,
            avg_re=self.avg_re,
            alive_nodes=self.alive_nodes)
        return cluster_heads

    def predict_cluster_assignments(self, cluster_heads, round):
        # Get the cluster assignments
        cluster_assignments = self.cluster_assignment_model.predict(
            network=self.network, cluster_heads=cluster_heads,
            std_re=self.std_re, std_el=self.std_el, round=round)
        # Now match each element in cluster assignments whose value is the index of the cluster head
        # to the actual cluster head
        # for i, cluster_head in enumerate(cluster_assignments):
        #     cluster_assignments[i] = cluster_heads[cluster_head]
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
            if node.remaining_energy <= 0:
                continue
            if cluster_head == 0:
                # if we reach this point, then the node  is alive but we may have made a bad prediction of the cluster head
                # Therefore, we assign the node to the closest cluster head
                min_distance = 10000
                ch_id = -1
                for ch in cluster_assignments:
                    if ch == 0:
                        continue
                    # Calculate the distance between the node and the cluster head
                    distance = self.network.distance_between_nodes(
                        node, self.network.get_node(ch))
                    if distance < min_distance:
                        min_distance = distance
                        ch_id = ch
                if ch_id == -1:
                    cluster_head = 1
                    # print(f"Node {node.node_id} is assigned to the sink.")
                    # print energy of node
                    # print(f"Node {node.node_id} energy: {node.remaining_energy}")
                    # print(f"alive nodes: {self.network.alive_nodes()}")
                else:
                    cluster_head = ch_id
            node.cluster_id = int(cluster_head)
            node.dst_to_cluster_head = self.network.distance_between_nodes(
                node, self.network.get_node(cluster_head))

    def evaluate_round(self, round):
        round += 1
        # print(f"Round {round}")

        self.re = self.network.remaining_energy()
        self.avg_re = self.network.average_remaining_energy()
        self.std_re = leach_surrogate.standardize_inputs(
            x=self.re, mean=self.cluster_head_model.re_mean,
            std=self.cluster_head_model.re_std)
        self.std_el = leach_surrogate.get_standardized_energy_levels(
            network=self.network,
            mean=self.cluster_head_model.el_mean,
            std=self.cluster_head_model.el_std)
        # print(f"Standardized energy levels: {self.std_el}")
        self.alive_nodes = self.network.alive_nodes()
        # Create cluster assignments predicted by the surrogate model
        cluster_heads = self.predict_cluster_heads(round=round)
        # Set cluster heads
        self.set_cluster_heads(cluster_heads)
        # Lets predict the cluster assignments
        cluster_assignments = self.predict_cluster_assignments(
            cluster_heads=cluster_heads, round=round)
        self.set_clusters(cluster_assignments)
        # with Timer() as t:
        self.net_model.dissipate_energy(round=round)
        # input("Press enter to continue...")
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
