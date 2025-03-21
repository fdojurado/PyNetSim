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


import copy


def copy_network(network: object, net_model: object):
    network_copy = copy.deepcopy(network)
    net_model_copy = copy.deepcopy(net_model)
    net_model_copy.set_network(network_copy)
    # Register callback to the network model
    # self.model.register_round_complete_callback(self.round_callback)
    # Register the callback to the network
    net_model_copy.register_round_complete_callback(
        network_copy.round_callback)
    return network_copy, net_model_copy


def dist_between_nodes(network: object, node1: int, node2: int):
    node1 = network.get_node(node1)
    node2 = network.get_node(node2)
    return network.distance_between_nodes(node1, node2)


def energy_spent_non_ch(network: object, src: int, dst: int):
    # src = network.get_node(src)
    # dst = network.get_node(dst)
    return network.calculate_energy_tx_non_ch(src=src, dst=dst)


def energy_spent_ch(network: object, src: int):
    # src = network.get_node(src)
    # print(f"CH: {src.node_id}, dst_to_sink: {src.dst_to_sink}")
    return network.calculate_energy_tx_ch(src=src)


def calculate_energy_ch_rx_per_node(network: object):
    return network.calculate_energy_rx_ch_per_node()


def target_load_balancing(network: object, ch: int,
                          a: float, b: float):
    ch = network.get_node(ch)
    ch_energy = ch.remaining_energy
    num_nodes = network.alive_nodes()
    return a * ch_energy + b * num_nodes


def update_cluster_heads(network: object, chs: list):
    for node in network:
        if node.node_id in chs:
            network.mark_as_cluster_head(
                node, node.node_id)
        else:
            network.mark_as_non_cluster_head(node)


def update_chs_to_nodes(network: object, assignments: dict):
    for node in assignments:
        src = network.get_node(int(node))
        ch = network.get_node(assignments[node])
        src.dst_to_cluster_head = dist_between_nodes(
            network, src.node_id, ch.node_id)
        src.cluster_id = ch.node_id


def get_energy(node: object):
    return node.remaining_energy
