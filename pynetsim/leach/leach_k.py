import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rich.progress import Progress
from sklearn.cluster import KMeans


class LEACH_K:

    def __init__(self, network, net_model: object):
        self.name = "LEACH-K"
        self.net_model = net_model
        self.config = network.config
        self.network = network

    @staticmethod
    def choose_cluster_heads(network: object, config: object):
        num_clusters = network.alive_nodes() * \
            config.network.protocol.cluster_head_percentage
        # Round up the number of clusters
        num_clusters = np.ceil(num_clusters)
        x = []
        y = []
        for node in network:
            if network.should_skip_node(node):
                continue
            x.append(node.x)
            y.append(node.y)
        coordinates = np.array(list(zip(x, y)))
        # print(f"Coordinates: {coordinates}")
        kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10)
        # fit the model to the coordinates
        kmeans.fit(coordinates)
        # get the cluster centers
        centers = kmeans.cluster_centers_
        # get the cluster labels
        labels = kmeans.labels_
        # print(f"Labels: {labels}")
        # Assign cluster ids to the nodes
        for node in network:
            if network.should_skip_node(node):
                continue
            # get index of node in coordinates
            index = np.where((coordinates[:, 0] == node.x) & (
                coordinates[:, 1] == node.y))
            # print(f"Node: {node.node_id}, index: {index}")
            # get label of node
            label = labels[index[0][0]]
            # print(f"Node: {node.node_id}, label: {label}")
            node.cluster_id = label

        # Assign cluster heads
        for cluster_id in range(int(num_clusters)):
            cluster_nodes = []
            for node in network:
                if network.should_skip_node(node):
                    continue
                if node.cluster_id == cluster_id:
                    cluster_nodes.append(node)
            # get the cluster head with the highest remaining energy
            cluster_head = max(
                cluster_nodes, key=lambda node: node.remaining_energy)
            # print(f"Cluster head: {cluster_head.node_id}")
            # set the cluster head
            network.mark_as_cluster_head(
                cluster_head, cluster_head.cluster_id)
        # Assign the distance to cluster head
        for node in network:
            if network.should_skip_node(node):
                continue
            if node.is_cluster_head:
                node.dst_to_cluster_head = node.dst_to_sink
            else:
                cluster_head = network.get_cluster_head(node)
                # print(f"Node {cluster_head.node_id} is cluster head with cluster id {cluster_head.cluster_id} of node {node.node_id}")
                node.dst_to_cluster_head = network.distance_between_nodes(
                    node, cluster_head)

        # visualize the clusters
        # nodes_labels = [node.cluster_id for node in self.network if not self.network.should_skip_node(
        #     node)]
        # plt.scatter(coordinates[:, 0], coordinates[:, 1],
        #             c=nodes_labels, s=50, cmap='viridis')
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        # # plot cluster heads
        # for node in self.network:
        #     if self.network.should_skip_node(node):
        #         continue
        #     if node.is_cluster_head:
        #         plt.scatter(node.x, node.y, c='red', s=200, alpha=0.5)
        # plt.show()
        # input("Press Enter to continue...")

    def run(self):
        print("Running LEACH_K...")
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

    def evaluate_round(self, round):
        round += 1

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        self.choose_cluster_heads()

        self.net_model.dissipate_energy(round=round)

        return round

    def run_without_plotting(self, num_rounds):
        round = 0
        with Progress() as progress:
            task = progress.add_task(
                "[red]Running LEACH_K...", total=num_rounds)
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
                print("Done!")
                ani.event_source.stop()

            ax.clear()
            common.plot_clusters(network=self.network, round=round, ax=ax)

            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
