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

import matplotlib.pyplot as plt
import json

# --------Plotting functions--------


def plot_clusters(network, round, ax):
    ax.clear()
    plot_nodes(network=network, ax=ax)
    plot_cluster_connections(network=network, ax=ax)
    annotate_node_ids(network=network, ax=ax)
    plot_sink_connections(network=network, ax=ax)
    ax.set_title(f"Round {round}")


def plot_nodes(network, ax):
    for node in network:
        if node.node_id == 1:
            node.color = "black"
        elif node.is_cluster_head and not node.is_main_cluster_head:
            node.color = "red"
        elif node.is_cluster_head and node.is_main_cluster_head:
            node.color = "green"
        else:
            node.color = "blue"
        ax.plot(node.x, node.y, 'o', color=node.color)


def plot_cluster_connections(network, ax):
    cluster_heads_exist = any(
        node.is_cluster_head for node in network)
    if not cluster_heads_exist:
        # print("There are no cluster heads.")
        return
    # print cluster heads
    chs = [node for node in network if node.is_cluster_head]
    # print(f"Cluster heads: {[ch.node_id for ch in chs]}")
    for node in network:
        if node.is_cluster_head or node.node_id == 1:
            continue
        # print(f"Node {node.node_id} is not a cluster head.")
        cluster_head = network.get_cluster_head(node=node)
        # if cluster_head:
        #     print(
        #         f"Node {node.node_id} is a member of cluster head {cluster_head.node_id}")
        ax.plot([node.x, cluster_head.x], [
                node.y, cluster_head.y], 'k--', linewidth=0.5)

    for node in network:
        if node.is_cluster_head and node.is_main_cluster_head:
            ax.plot([node.x, network.nodes[1].x], [
                node.y, network.nodes[1].y], 'k-', linewidth=1)
        elif node.is_cluster_head and node.mch_id != 0:
            mch = network.get_mch(node=node)
            ax.plot([node.x, mch.x], [node.y, mch.y], 'k-', linewidth=0.5)
        elif node.is_cluster_head:
            ax.plot([node.x, network.nodes[1].x], [
                node.y, network.nodes[1].y], 'k-', linewidth=1)


def plot_sink_connections(network, ax):
    for node in network:
        if node.node_id == 1:
            ax.plot([node.x, network.nodes[1].x], [
                    node.y, network.nodes[1].y], 'k-', linewidth=1)


def annotate_node_ids(network, ax):
    for node in network:
        ax.annotate(node.node_id, (node.x, node.y))


def plot_metrics(network_energy, network_energy_label, network_energy_unit,
                 network_energy_title, num_dead_nodes, num_dead_nodes_label,
                 num_dead_nodes_title,
                 num_alive_nodes, num_alive_nodes_label, num_alive_nodes_title):
    plt.figure()
    plt.plot(network_energy.keys(), network_energy.values())
    plt.xlabel("Round")
    plt.ylabel(f"{network_energy_label} ({network_energy_unit})")
    plt.title(network_energy_title)
    plt.show()

    plt.figure()
    plt.plot(num_dead_nodes.keys(), num_dead_nodes.values())
    plt.xlabel("Round")
    plt.ylabel(num_dead_nodes_label)
    plt.title(num_dead_nodes_title)
    plt.show()

    plt.figure()
    plt.plot(num_alive_nodes.keys(), num_alive_nodes.values())
    plt.xlabel("Round")
    plt.ylabel(num_alive_nodes_label)
    plt.title(num_alive_nodes_title)
    plt.show()
