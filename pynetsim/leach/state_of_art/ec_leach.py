# This is based on:
# @article{bsoul2013energy,
#   title={An energy-efficient threshold-based clustering protocol for wireless sensor networks},
#   author={Bsoul, Mohammad and Al-Khasawneh, Ahmad and Abdallah, Alaa E and Abdallah, Emad E and Obeidat, Ibrahim},
#   journal={Wireless personal communications},
#   volume={70},
#   pages={99--112},
#   year={2013},
#   publisher={Springer}
#   url={https://link.springer.com/content/pdf/10.1007/s11277-012-0681-8.pdf}
# }
import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rich.progress import Progress


class EC_LEACH:

    def __init__(self, network, net_model: object):
        self.name = "EC-LEACH"
        self.net_model = net_model
        self.config = network.config
        self.network = network

    def cluster_heads_candidates(self):
        threshold = {}
        for node in self.network:
            sum = 0
            for other_node in self.network:
                if node.node_id == other_node.node_id:
                    continue
                sum += self.network.distance_between_nodes(node, other_node) / \
                    other_node.remaining_energy
            threshold[node.node_id] = node.remaining_energy / sum
        return threshold

    def choose_cluster_heads(self, sorted_ch_candidates):
        ch = []
        for node_id in sorted_ch_candidates:
            # check if the ch list is empty
            if len(ch) == 0:
                ch.append(node_id)
                continue
            # Get the last element of the list
            last_ch = ch[-1]
            if self.network.distance_between_nodes(self.network.get_node(node_id), self.network.get_node(last_ch)) >= 30:
                ch.append(node_id)
            else:
                continue
            if len(ch) >= self.max_chs:
                break
        # Mark the nodes as cluster heads
        for node_id in ch:
            node = self.network.get_node(node_id)
            self.network.mark_as_cluster_head(node, node_id)
        print(f"Cluster heads: {ch}")

    def run(self):
        print(f"Running {self.name}...")
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = self.config.network.plot
        plot_refresh = self.config.network.plot_refresh

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
                num_rounds, plot_refresh)

    def evaluate_round(self, round):
        round += 1

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        self.max_chs = np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage)

        ch_candidates = self.cluster_heads_candidates()

        # print(f"ch candidates: {ch_candidates}")

        # Sort the dictionary by value in descending order
        sorted_ch_candidates = sorted(
            ch_candidates, key=ch_candidates.get, reverse=True)

        # print(f"Sorted ch candidates: {sorted_ch_candidates}")

        self.choose_cluster_heads(sorted_ch_candidates)
        self.network.create_clusters()
        self.net_model.dissipate_energy(round=round)

        return round

    def run_without_plotting(self, num_rounds):
        round = 0
        with Progress() as progress:
            task = progress.add_task(
                "[red]Running LEACH_C...", total=num_rounds)
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

            plt.pause(plot_refresh)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(0, num_rounds + 1), repeat=False)

        plt.show()
