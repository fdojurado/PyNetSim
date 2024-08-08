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

# This is based on:
# @article{parmar_improved_2015,
# 	title = {An improved modiﬁed {LEACH}-{C} algorithm for energy efﬁcient routing in {Wireless} {Sensor} {Networks}},
# 	volume = {4},
# 	abstract = {Wireless Sensor Network (WSN) is mainly characterized by its limited power supply. Hence, protocols designed for WSNs should be energy efﬁcient. Cluster based routing helps to improve the network lifetime. Centralized Low-Energy Adaptive Clustering Hierarchy (LEACH-C) is an energy efﬁcient cluster based routing protocol that has shown improvement over Low Energy Adaptive Clustering Heirarchy (LEACH) protocol. A modiﬁed LEACH-C (LEACH-CM) protocol is proposed in this paper that considers the distance between the selected cluster head (CH) and a member node; and the distance between the member node and Base Station (BS) to transmit data by a member node to CH or BS. The proposed approach selects number of CHs on the basis of alive nodes in the network, rather than considering total nodes in the network. Simulation results show that LEACH-CM outperforms to LEACH-C, and improves the network lifetime.},
# 	language = {en},
# 	author = {Parmar, Amit and Thakkar, Ankit},
# 	year = {2015},
# 	file = {Parmar and Thakkar - 2015 - An improved modiﬁed LEACH-C algorithm for energy e.pdf:/Users/fernando/Zotero/storage/SFB9GRWR/Parmar and Thakkar - 2015 - An improved modiﬁed LEACH-C algorithm for energy e.pdf:application/pdf},
# }

from pynetsim.leach.leach_c import LEACH_C
import numpy as np
import math


class LEACH_CM(LEACH_C):

    def __init__(self, network, net_model):
        self.name = "LEACH-CM"
        super().__init__(network=network, net_model=net_model, name=self.name)

    def simulated_annealing(self, cluster_heads):
        number_of_cluster_heads = self.network.alive_nodes(
        )*self.config.network.protocol.cluster_head_percentage
        if number_of_cluster_heads < 1:
            number_of_cluster_heads = math.ceil(number_of_cluster_heads)
        else:
            number_of_cluster_heads = math.floor(number_of_cluster_heads)
        number_of_cluster_heads = int(number_of_cluster_heads)
        # number_of_cluster_heads = int(self.network.alive_nodes(
        # )*self.config.network.protocol.cluster_head_percentage)+1
        # if number_of_cluster_heads < 1:
        #     number_of_cluster_heads = 1
        if len(cluster_heads) < number_of_cluster_heads:
            number_of_cluster_heads = len(cluster_heads)
        best = self.rng.get_np_random_choice(
            cluster_heads, size=number_of_cluster_heads, replace=False)

        best_eval = self.objective_function(best)
        curr, curr_eval = best, best_eval

        initial_temp = 10

        for i in range(100):

            candidate = self.rng.get_np_random_choice(
                cluster_heads, size=number_of_cluster_heads, replace=False)

            candidate_eval = self.objective_function(candidate)

            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval

            diff = candidate_eval - curr_eval

            t = initial_temp / (i + 1)
            metropolis = np.exp(-diff / t)
            if diff < 0 or self.rng.get_random() < metropolis:
                curr, curr_eval = candidate, candidate_eval

        for node in self.network:
            if node.node_id in best:
                self.num_cluster_heads += 1
                self.network.mark_as_cluster_head(
                    node, self.num_cluster_heads)

    def create_clusters(self):
        cluster_heads_exist = any(
            node.is_cluster_head for node in self.network)
        if not cluster_heads_exist:
            self.network.clear_clusters()
            return False

        for node in self.network:
            if not node.is_cluster_head and node.node_id != 1:
                self.add_node_to_cluster(node=node)

        return True

    def add_node_to_cluster(self, node):
        # Get all cluster heads including the base station
        cluster_heads = [
            cluster_head for cluster_head in self.network if cluster_head.is_cluster_head]
        # Add the BS station to the list of cluster heads
        cluster_heads.append(self.network.get_node(1))
        distances = {cluster_head.node_id: ((node.x - cluster_head.x)**2 + (node.y - cluster_head.y)**2)**0.5
                     for cluster_head in cluster_heads}
        cluster_head_id = min(distances, key=distances.get)
        min_distance = distances[cluster_head_id]
        cluster_head = self.network.get_node(cluster_head_id)
        cluster_head.add_neighbor(node)
        node.add_neighbor(cluster_head)
        node.dst_to_cluster_head = min_distance
        node.cluster_id = cluster_head.cluster_id

    def evaluate_round(self, round):
        round += 1

        # print(f"Evaluating round {round} of {self.name}")

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        self.num_cluster_heads = 0

        network_avg_energy = self.network.average_remaining_energy()

        potential_chs = self.potential_cluster_heads(network_avg_energy)
        # print(f"Potential cluster heads: {potential_chs}")

        self.choose_cluster_heads(potential_chs)
        # print cluster heads
        # cluster_heads = [
        #     node for node in self.network if node.is_cluster_head]
        # print(f"Cluster heads: {[node.node_id for node in cluster_heads]}")
        self.create_clusters()
        self.net_model.dissipate_energy(round=round)

        return round
