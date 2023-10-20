import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pyomo.environ as pyo
import pynetsim.leach.leach_milp as leach_milp


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
        self.alpha = 3.3
        self.beta = 0.9
        self.gamma = 1.7
        self.surrogate_model = SurrogateModel(config=self.config)

    def run(self):
        logger.info(
            f"Running {self.name} with alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}")
        self.metrics = {}
        self.model, _, _ = self.surrogate_model.get_model()
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

    def get_network_metrics(self):

        weights = [self.alpha/self.largest_weight, self.beta /
                   self.largest_weight, self.gamma/self.largest_weight]

        assert all(
            0 <= w <= 1 for w in weights), f"Incorrect weights: {weights}"

        # logger.info(f"Weights: {w_np}, shape: {w_np.shape}")
        # print all energy levels
        # for node in self.network:
        #     logger.info(f"Node {node.node_id} energy level: {node.remaining_energy}")
        # We need to get energy levels for all nodes
        energy_levels = [
            node.remaining_energy for node in self.network if node.node_id != 1]
        assert all(-1 <= e <=
                   1 for e in energy_levels), f"Incorrect energy levels: {energy_levels}"
        # logger.info(
        #     f"Energy levels: {energy_levels}, shape: {energy_levels.shape}")
        remaining_energy = self.network.remaining_energy()/100
        assert 0 <= remaining_energy <= 1, f"Incorrect remaining energy: {remaining_energy}"

        alive_nodes = self.network.alive_nodes()/100
        assert 0 <= alive_nodes <= 1, f"Incorrect alive nodes: {alive_nodes}"

        num_chs = self.network.num_cluster_heads()/5
        assert 0 <= num_chs <= 1, f"Incorrect num_chs: {num_chs}"

        control_packets_energy = self.network.control_packets_energy()/5
        assert 0 <= control_packets_energy <= 1, f"Incorrect control_packets_energy: {control_packets_energy}"

        control_pkt_bits = self.network.control_pkt_bits()/1e8
        assert 0 <= control_pkt_bits <= 1, f"Incorrect control_pkt_bits: {control_pkt_bits}"

        pkts_sent_to_bs = self.network.pkts_sent_to_bs()/1e3
        assert 0 <= pkts_sent_to_bs <= 1, f"Incorrect pkts_sent_to_bs: {pkts_sent_to_bs}"

        pkts_recv_by_bs = self.network.pkts_recv_by_bs()/1e3
        assert 0 <= pkts_recv_by_bs <= 1, f"Incorrect pkts_recv_by_bs: {pkts_recv_by_bs}"

        energy_dissipated = self.network.energy_dissipated()/100
        assert 0 <= energy_dissipated <= 1, f"Incorrect energy_dissipated: {energy_dissipated}"

        # We need to get the current membership of all nodes, this involves in getting the
        # cluster id of all nodes.
        cluster_ids = [0 if node.cluster_id is None else node.cluster_id /
                       100 for node in self.network if node.node_id != 1]
        assert all(
            0 <= c <= 1 for c in cluster_ids), f"Incorrect cluster ids: {cluster_ids}"

        x_data = weights + energy_levels + [remaining_energy, alive_nodes, num_chs, control_packets_energy,
                                            control_pkt_bits, pkts_sent_to_bs, pkts_recv_by_bs, energy_dissipated] + cluster_ids

        return x_data

    def save_metrics(self, round):
        network_metrics = self.get_network_metrics()
        self.metrics[round] = network_metrics

    def predict_cluster_heads(self, round):
        logger.info(f"Predicting cluster heads for round {round}...")
        # Before we can predict the cluster assignments, we need to create the input data
        # for the surrogate model.
        if round == 1:
            self.save_metrics(round-1)
            # input(f"X data: {x_data}")
        x_data = self.metrics[round - 1]
        # print(f"X data: {x_data}")
        prev_x_data = []
        for prev_round in range(round-2, round-12, -1):
            # input(f"Round: {round}, prev round: {prev_round}")
            if prev_round < 0:
                prev_round_data = [0 for _ in range(len(x_data))]
            else:
                prev_round_data = self.metrics[prev_round]
            prev_x_data += prev_round_data
        # Convert to numpy array
        np_x = np.array(x_data)
        # print(f"X data: {np_x}, shape: {np_x.shape}")
        np_prev_x = np.array(prev_x_data)
        # print(f"Prev X data: {np_prev_x}, shape: {np_prev_x.shape}")
        # Concatenate the two arrays
        np_x = np.concatenate((np_x, np_prev_x))
        # print(f"X data: {np_x}, shape: {np_x.shape}")
        # print shapes
        # Convert the numerical data to a tensor
        x_data_tensor = torch.from_numpy(np_x.astype(np.float32))
        # Print the X every 209 elements in the array
        # for i in range(0, len(x_data_tensor), 209):
        #     print(f"X data {i/209}: {x_data_tensor[i:i+209]}")
        # print(f"X data tensor: {x_data_tensor.tolist()}, shape: {x_data_tensor.shape}")
        # unsqueeze the tensor
        x_data_tensor = x_data_tensor.unsqueeze(0)

        self.model.eval()
        # logger.info(
        #     f"Shapes: {numerical_data.shape}, {categorical_data.shape}")
        with torch.no_grad():
            # logger.info(
            #     f"Categorical data: {categorical_data}, shape: {categorical_data.shape}")
            output = self.model(x_data_tensor)
            # logger.info(f"Output: {output}, shape: {output.shape}")
            _, predicted_assignments = torch.max(output.data[0], 1)
            # logger.info(
            # f"Predicted cluster heads: {predicted_assignments}")
        # Convert the predicted assignments to a numpy array
        predicted_assignments = predicted_assignments.numpy()
        # print(f"Predicted cluster heads: {predicted_assignments}")
        return predicted_assignments

    def set_clusters(self, cluster_assignments):
        # print(f"Cluster assignments: {cluster_assignments}")
        # Get unique cluster ids
        unique_cluster_ids = np.unique(cluster_assignments)
        # print(
        #     f"Unique cluster ids: {unique_cluster_ids}, len: {len(unique_cluster_ids)}")
        # See how many times each cluster id appears in the cluster assignments
        cluster_id_counts = np.bincount(cluster_assignments)
        # print(
        #     f"Cluster id counts: {cluster_id_counts}, len: {len(cluster_id_counts)}")
        # Lets create a dictionary of cluster ids and their counts
        cluster_id_counts_dict = {}
        for unique_cluster_id in unique_cluster_ids:
            cluster_id_counts_dict[unique_cluster_id] = cluster_id_counts[unique_cluster_id]
        # print(f"Cluster id counts: {cluster_id_counts_dict}")
        # Lets sort the dictionary by the counts
        sorted_cluster_id_counts_dict = {k: v for k, v in sorted(
            cluster_id_counts_dict.items(), key=lambda item: item[1], reverse=True)}
        # print(
        #     f"Sorted cluster id counts: {sorted_cluster_id_counts_dict}, len: {len(sorted_cluster_id_counts_dict)}")
        # Remove nodes from the cluster_id_counts_dict that have their energy below the network average remaining energy
        network_avg_energy = self.network.average_remaining_energy()
        # print(f"Network average energy: {network_avg_energy}")
        # loop through the cluster_id_counts_dict and remove nodes that have their energy below the network average remaining energy
        final_cluster_id = {}
        for cluster_id, count in sorted_cluster_id_counts_dict.items():
            if cluster_id == 0:
                continue
            node = self.network.get_node(cluster_id)
            # if node.remaining_energy < network_avg_energy:
            #     print(
            #         f"Node {node.node_id} has energy {node.remaining_energy} below the network average energy. Removing from cluster id counts.")
            #     continue
            final_cluster_id[cluster_id] = {}
            final_cluster_id[cluster_id]["count"] = count
            final_cluster_id[cluster_id]["energy"] = node.remaining_energy
        # print(
        #     f"Sorted cluster id counts after removing nodes with energy below the network average energy: {final_cluster_id}, len: {len(final_cluster_id)}")
        # Lets set the cluster heads which are the first self.max_chs in the final_cluster_id
        # Sort the final_cluster_id by remaining energy
        final_cluster_id = {k: v for k, v in sorted(
            final_cluster_id.items(), key=lambda item: item[1]["energy"], reverse=True)}
        # print(
        #     f"Sorted final cluster id counts by remaining energy: {final_cluster_id}, len: {len(final_cluster_id)}")
        cluster_heads = list(final_cluster_id.keys())[:int(self.max_chs)]
        # Sort the cluster heads
        cluster_heads.sort()
        print(f"Cluster heads: {cluster_heads}")
        # Lets get those cluster heads that were not selected
        cluster_heads_not_selected = list(final_cluster_id.keys())[
            int(self.max_chs):]
        print(f"Cluster heads not selected: {cluster_heads_not_selected}")
        for node in self.network:
            if node.node_id in cluster_heads and node.node_id != 1:
                self.network.mark_as_cluster_head(node, node.node_id)
        # Assign the cluster ids to the nodes
        for node in self.network:
            if node.node_id == 1:
                continue
            if node.remaining_energy < 0:
                node.cluster_id = 0
                continue
            if node.node_id not in cluster_heads:
                self.network.mark_as_non_cluster_head(node)
                cluster_head_id = int(cluster_assignments[node.node_id-2])
                if cluster_head_id in cluster_heads_not_selected:
                    print(
                        f"Node {node.node_id} is assigned to a cluster head {cluster_head_id} that was not selected.")
                    # Select the closest cluster head
                    min_distance = 10000
                    ch_id = -1
                    for ch in cluster_heads:
                        # Calculate the distance between the node and the cluster head
                        distance = self.network.distance_between_nodes(
                            self.network.get_node(node.node_id), self.network.get_node(ch))
                        if distance < min_distance:
                            min_distance = distance
                            ch_id = ch
                    cluster_head_id = int(ch_id)
                    print(
                        f"Node {node.node_id} is assigned to cluster head {cluster_head_id}")
                node.cluster_id = cluster_head_id
                if cluster_head_id != 0:
                    cluster_head = self.network.get_node(cluster_head_id)
                    node.dst_to_cluster_head = self.network.distance_between_nodes(
                        node, cluster_head)
        # for node in self.network:
        #     if self.network.should_skip_node(node):
        #         continue
        #     if node.node_id in cluster_assignments:
        #         self.network.mark_as_cluster_head(node, node.node_id)

    def print_clusters(self):
        # Print cluster assignments
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            logger.info(
                f"Node {node.node_id} cluster head: {node.cluster_id}")

    def generate_initial_cluster_assignments(self):
        # Select num_chs cluster heads from the network
        # chs = np.random.choice(
        #     [node.node_id for node in self.network if node.node_id != 1], size=int(self.num_chs), replace=False)
        # logger.info(f"Cluster heads: {chs}")
        # # convert chs to int
        # chs = [int(ch) for ch in chs]
        # # Set them as cluster heads
        # for node_id in chs:
        #     node = self.network.get_node(node_id)
        #     self.network.mark_as_cluster_head(node, node_id)
        # # Create clusters
        # self.network.create_clusters()
        chs = [2, 14, 98, 21, 58]
        # Mark them as cluster heads
        for cluster_id in chs:
            node = self.network.get_node(cluster_id)
            self.network.mark_as_cluster_head(node, cluster_id)
        cluster_assignments = {
            "2": 2, "3": 14, "4": 98, "5": 21, "6": 2, "7": 21, "8": 14, "9": 14, "10": 98, "11": 21, "12": 98, "13": 21, "14": 14, "15": 2, "16": 14, "17": 14, "18": 14, "19": 21, "20": 21, "21": 21, "22": 2, "23": 58, "24": 2, "25": 2, "26": 21, "27": 21, "28": 58, "29": 2, "30": 58, "31": 14, "32": 21, "33": 14, "34": 21, "35": 14, "36": 98, "37": 14, "38": 21, "39": 14, "40": 2, "41": 58, "42": 14, "43": 58, "44": 14, "45": 14, "46": 21, "47": 21, "48": 14, "49": 21, "50": 58, "51": 98, "52": 98, "53": 2, "54": 14, "55": 21, "56": 58, "57": 21, "58": 58, "59": 2, "60": 2, "61": 14, "62": 21, "63": 21, "64": 14, "65": 21, "66": 2, "67": 98, "68": 98, "69": 14, "70": 98, "71": 14, "72": 14, "73": 21, "74": 21, "75": 14, "76": 14, "77": 2, "78": 2, "79": 21, "80": 21, "81": 21, "82": 2, "83": 14, "84": 14, "85": 21, "86": 58, "87": 98, "88": 2, "89": 21, "90": 58, "91": 98, "92": 21, "93": 98, "94": 21, "95": 14, "96": 21, "97": 14, "98": 98, "99": 98, "100": 21
        }
        for node in self.network:
            if node.node_id not in chs and node.node_id != 1:
                self.network.mark_as_non_cluster_head(node)
                cluster_head_id = cluster_assignments[str(node.node_id)]
                cluster_head = self.network.get_node(cluster_head_id)
                node.cluster_id = cluster_head_id
                node.dst_to_cluster_head = self.network.distance_between_nodes(
                    node, cluster_head)

    def generate_initial_cluster_assignments_two(self):
        chs = [6, 95, 67, 57, 90]
        # Mark them as cluster heads
        for cluster_id in chs:
            node = self.network.get_node(cluster_id)
            self.network.mark_as_cluster_head(node, cluster_id)
        cluster_assignments = {
            "2": 6, "3": 95, "4": 67, "5": 95, "6": 6, "7": 57, "8": 90, "9": 95, "10": 67, "11": 57, "12": 67, "13": 57, "14": 95, "15": 6, "16": 95, "17": 90, "18": 95, "19": 57, "20": 57, "21": 57, "22": 6, "23": 90, "24": 6, "25": 6, "26": 57, "27": 57, "28": 90, "29": 6, "30": 90, "31": 95, "32": 95, "33": 95, "34": 57, "35": 95, "36": 67, "37": 95, "38": 57, "39": 95, "40": 67, "41": 90, "42": 95, "43": 90, "44": 90, "45": 95, "46": 57, "47": 57, "48": 95, "49": 57, "50": 90, "51": 67, "52": 67, "53": 6, "54": 95, "55": 57, "56": 90, "57": 57, "58": 90, "59": 6, "60": 6, "61": 95, "62": 95, "63": 57, "64": 90, "65": 57, "66": 6, "67": 67, "68": 67, "69": 90, "70": 67, "71": 90, "72": 95, "73": 57, "74": 95, "75": 95, "76": 95, "77": 6, "78": 6, "79": 57, "80": 57, "81": 57, "82": 6, "83": 90, "84": 95, "85": 57, "86": 90, "87": 67, "88": 6, "89": 57, "90": 90, "91": 67, "92": 57, "93": 67, "94": 57, "95": 95, "96": 57, "97": 95, "98": 67, "99": 67, "100": 57
        }
        for node in self.network:
            if node.node_id not in chs and node.node_id != 1:
                self.network.mark_as_non_cluster_head(node)
                cluster_head_id = cluster_assignments[str(node.node_id)]
                cluster_head = self.network.get_node(cluster_head_id)
                node.cluster_id = cluster_head_id
                node.dst_to_cluster_head = self.network.distance_between_nodes(
                    node, cluster_head)

    def evaluate_round(self, round):
        round += 1
        print(f"Round {round}")

        self.max_chs = int(self.network.alive_nodes() *
                           self.config.network.protocol.cluster_head_percentage) + 1

        # if round == 1:
        #     for node in self.network:
        #         self.network.mark_as_non_cluster_head(node)

        #     # We need to get the initial cluster assignments
        #     self.generate_initial_cluster_assignments()
        # elif round == 2:
        #     for node in self.network:
        #         self.network.mark_as_non_cluster_head(node)

        # We need to get the initial cluster assignments
        # self.generate_initial_cluster_assignments_two()
        # else:
        # Create cluster assignments predicted by the surrogate model
        cluster_assignments = self.predict_cluster_heads(round=round)
        for node in self.network:
            self.network.mark_as_non_cluster_head(node)
        self.set_clusters(cluster_assignments)
        # self.network.create_clusters()
        # self.print_clusters()
        # print(f"Cluster heads at round {round}: {chs}")
        # print(f"Cluster assignments at round {round}: {node_cluster_head}")
        # Save the metrics
        self.net_model.dissipate_energy(round=round)
        self.save_metrics(round=round)
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
