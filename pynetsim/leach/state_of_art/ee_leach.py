# This is based on:
# @inproceedings{tripathi2013energy,
#   title={Energy efficient clustered routing for wireless sensor network},
#   author={Tripathi, Meenakshi and Battula, Ramesh Babu and Gaur, Manoj Singh and Laxmi, Vijay},
#   booktitle={2013 IEEE 9th International Conference on Mobile Ad-hoc and Sensor Networks},
#   pages={330--335},
#   year={2013},
#   organization={IEEE}
#   url={https://ieeexplore.ieee.org/iel7/6724377/6726286/06726352.pdf}
# }
import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rich.progress import Progress
from pynetsim.utils import RandomNumberGenerator


class EE_LEACH:

    def __init__(self, network, net_model: object):
        self.name = "EE-LEACH"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.rng = RandomNumberGenerator(self.config)

    def cluster_heads_candidates(self):
        # Average energy of the network
        network_avg_energy = self.network.average_remaining_energy()

        self.max_chs = int(np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage))

        # print(f"Max CHs: {self.max_chs}")

        # get all nodes above the average energy
        potential_cluster_heads = []
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            if node.remaining_energy >= network_avg_energy:
                potential_cluster_heads.append(node.node_id)

        # print(f"Potential cluster heads: {potential_cluster_heads}")

        # Select a random population of size max_chs from the potential cluster heads
        # return np.random.choice(potential_cluster_heads, self.max_chs, replace=False)
        return self.rng.get_np_random_choice(potential_cluster_heads, self.max_chs, replace=False)

    def node_distance_to_cluster_candidate(self, node,
                                           cluster_head_assignments):
        # Calculate the distance of node to each cluster head and return the minimum
        cluster_heads = [
            self.network.get_node(cluster_head_id)
            for cluster_head_id in cluster_head_assignments
        ]
        distances = {
            cluster_head.node_id: (
                self.network.distance_between_nodes(node, cluster_head))
            for cluster_head in cluster_heads
        }
        cluster_head_id = min(distances, key=distances.get)
        return cluster_head_id

    def assign_cluster_members(self, ch_candidates):
        # Now, non-CHs will choose the closest CH
        cluster_members = {}
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            if node.node_id in ch_candidates:
                # Skip cluster heads
                continue
            # Assign cluster head
            ch_id = self.node_distance_to_cluster_candidate(
                node, ch_candidates)
            # print(f"Node {node.node_id} is assigned to {ch_id}")
            cluster_members[node.node_id] = ch_id

        # print(f"Cluster members: {cluster_members}")

        # print per cluster head
        for ch_id in ch_candidates:
            nodes = [
                node_id for node_id in cluster_members if cluster_members[node_id] == ch_id]
            # print(f"Cluster head {ch_id} has nodes: {nodes}")
            pass

        return cluster_members

    def choose_cluster_heads(self, ch_candidates, cluster_members):
        # Choose as cluster the node with the highest residual energy
        chs = []
        for ch_candidate in ch_candidates:
            # Get the nodes assigned to this cluster head
            nodes = [
                node_id for node_id in cluster_members if cluster_members[node_id] == ch_candidate]
            # Also add the ch_candidate to the list
            nodes.append(ch_candidate)
            if len(nodes) == 0:
                continue
            # Get the node with the highest residual energy
            max_energy = 0
            max_node = None
            # print("Energies: ", end="")
            for node_id in nodes:
                node = self.network.get_node(node_id)
                # print(f"{node.node_id}={node.remaining_energy:.2f} ", end="")
                if node.remaining_energy > max_energy:
                    max_energy = node.remaining_energy
                    max_node = node
            # print()
            if max_node is not None:
                chs.append(max_node.node_id)
            # print(f"Node {max_node.node_id} is a cluster head.")
            # Mark the max_node as cluster head
            # print(f"Node {max_node.node_id} is a cluster head.")
            self.network.mark_as_cluster_head(max_node, max_node.node_id)
            # Set the memberships for each non-cluster head node
            for node_id in nodes:
                if node_id == max_node.node_id:
                    continue
                node = self.network.get_node(node_id)
                node.dst_to_cluster_head = self.network.distance_between_nodes(
                    node, max_node)
                node.cluster_head = max_node.node_id

        # print(f"Cluster heads: {chs}")

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

        # Assign cluster members
        cluster_members = self.assign_cluster_members(ch_candidates)

        # print(f"Sorted ch candidates: {sorted_ch_candidates}")

        self.choose_cluster_heads(ch_candidates, cluster_members)
        self.net_model.dissipate_energy(round=round)

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

    def run_with_plotting(self, num_rounds, plot_refresh):
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
