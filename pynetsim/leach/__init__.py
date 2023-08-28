import matplotlib.pyplot as plt
import json


def energy_dissipation_non_cluster_heads(round, network,
                                         elect, eda, packet_size, eamp):
    # print("Energy dissipation for non-cluster heads")
    for node in network.nodes.values():
        if should_skip_node(node) or node.is_cluster_head:
            continue

        cluster_head = get_cluster_head(network=network, node=node)
        if cluster_head is None:
            transfer_data_to_sink(network=network, node=node,
                                  elect=elect, packet_size=packet_size,
                                  eamp=eamp, round=round)
        else:
            process_non_cluster_head(network=network, node=node,
                                     cluster_head=cluster_head, round=round,
                                     elect=elect, eda=eda, packet_size=packet_size,
                                     eamp=eamp)


def energy_dissipation_cluster_heads(round, network,
                                     elect, eda, packet_size, eamp):
    for node in network.nodes.values():
        if not node.is_cluster_head or node.energy <= 0:
            continue
        distance = node.dst_to_sink
        # print(
        #     f"Cluster head {node.node_id} with cluster id {node.cluster_id} distance to sink: {distance}")
        ETx = (elect + eda) * packet_size + \
            eamp * packet_size * distance**2
        node.energy -= ETx
        # network.remaining_energy -= ETx
        if node.energy <= 0:
            mark_node_as_dead(node, round)
            remove_cluster_head_from_cluster(
                network=network, cluster_head=node)


def transfer_data_to_sink(network, node, elect, packet_size, eamp, round):
    distance = node.dst_to_sink
    # print("No cluster heads, transferring data to the sink.")
    # print(f"Node {node.node_id} distance to sink: {distance}")
    ETx = elect * packet_size + eamp * packet_size * distance**2
    node.energy -= ETx
    # network.remaining_energy -= ETx
    if node.energy <= 0:
        mark_node_as_dead(node, round)


def process_non_cluster_head(network, node, cluster_head, round,
                             elect, eda, packet_size, eamp):
    distance = get_node_distance(node, cluster_head)
    # print(f"Node {node.node_id} distance to cluster head: {distance}")
    ETx = calculate_tx_energy_dissipation(distance=distance, elect=elect,
                                          packet_size=packet_size, eamp=eamp)
    node.energy -= ETx
    # network.remaining_energy -= ETx
    ERx = (elect + eda) * packet_size
    cluster_head.energy -= ERx
    # network.remaining_energy -= ERx
    if cluster_head.energy <= 0:
        # print(f"Cluster head {cluster_head.node_id} is dead.")
        mark_node_as_dead(cluster_head, round)
        remove_cluster_head(network=network, cluster_head=cluster_head)
        remove_node_from_cluster(cluster_head)
        remove_cluster_head_from_cluster(network=network,
                                         cluster_head=cluster_head)
    if node.energy <= 0:
        print(f"Node {node.node_id} is dead.")
        mark_node_as_dead(node, round)
        remove_node_from_cluster(node)


def create_clusters(network):
    cluster_heads_exist = any(
        node.is_cluster_head for node in network.nodes.values())
    if not cluster_heads_exist:
        print("There are no cluster heads.")
        # input("Press Enter to continue...")
        clear_clusters(network)
        return False

    for node in network.nodes.values():
        if not node.is_cluster_head and node.node_id != 1:
            add_node_to_cluster(network=network, node=node)


def add_node_to_cluster(network, node):
    distances = {cluster_head.node_id: ((node.x - cluster_head.x)**2 + (node.y - cluster_head.y)**2)**0.5
                 for cluster_head in network.nodes.values() if cluster_head.is_cluster_head}
    cluster_head_id = min(distances, key=distances.get)
    min_distance = distances[cluster_head_id]
    cluster_head = network.nodes[cluster_head_id]
    cluster_head.add_neighbor(node)
    node.add_neighbor(cluster_head)
    node.dst_to_cluster_head = min_distance
    node.cluster_id = cluster_head.cluster_id
    # print(
    #     f"Node {node.node_id} is in the cluster of node {cluster_head_id}.")


def mark_as_cluster_head(network, node, num_cluster_heads):
    num_cluster_heads += 1
    node.is_cluster_head = True
    node.cluster_id = num_cluster_heads
    # print(f"Node {node.node_id} is cluster head")
    return num_cluster_heads


def store_metrics(config, network, round, network_energy, num_dead_nodes, num_alive_nodes):
    num_nodes = config.network.num_sensor
    network_energy[round] = network.remaining_energy()
    num_dead_nodes[round] = num_nodes - network.alive_nodes()
    num_alive_nodes[round] = network.alive_nodes()


def plot_clusters(network, round, ax):
    ax.clear()
    plot_nodes(network=network, ax=ax)
    plot_cluster_connections(network=network, ax=ax)
    annotate_node_ids(network=network, ax=ax)
    plot_sink_connections(network=network, ax=ax)
    ax.set_title(f"Round {round}")


def plot_nodes(network, ax):
    for node in network.nodes.values():
        if node.node_id == 1:
            node.color = "black"
        elif node.is_cluster_head:
            node.color = "red"
        else:
            node.color = "blue"
        ax.plot(node.x, node.y, 'o', color=node.color)


def plot_cluster_connections(network, ax):
    cluster_heads_exist = any(
        node.is_cluster_head for node in network.nodes.values())
    if not cluster_heads_exist:
        print("There are no cluster heads.")
        return
    for node in network.nodes.values():
        if node.is_cluster_head or node.node_id == 1:
            continue
        cluster_head = get_cluster_head(network=network, node=node)
        ax.plot([node.x, cluster_head.x], [
                node.y, cluster_head.y], 'k--', linewidth=0.5)

    for node in network.nodes.values():
        if node.is_cluster_head:
            ax.plot([node.x, network.nodes[1].x], [
                    node.y, network.nodes[1].y], 'k-', linewidth=1)


def plot_sink_connections(network, ax):
    for node in network.nodes.values():
        if node.node_id == 1:
            ax.plot([node.x, network.nodes[1].x], [
                    node.y, network.nodes[1].y], 'k-', linewidth=1)


def annotate_node_ids(network, ax):
    for node in network.nodes.values():
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


def should_skip_node(node):
    return node.node_id == 1 or node.energy <= 0


def get_cluster_head(network, node):
    return network.get_node_with_cluster_id(node.cluster_id)


def get_node_distance(node, cluster_head):
    return node.dst_to_cluster_head if cluster_head.energy > 0 else node.dst_to_sink


def calculate_tx_energy_dissipation(distance, elect, packet_size, eamp):
    return elect * packet_size + eamp * packet_size * distance**2


def clear_clusters(network):
    for node in network.nodes.values():
        node.cluster_id = 0


def remove_cluster_head(network, cluster_head):
    cluster_id = cluster_head.cluster_id
    for node in network.nodes.values():
        if node.cluster_id == cluster_id:
            node.cluster_id = 0


def remove_node_from_cluster(node):
    for neighbor in node.neighbors.values():
        # print(f"Removing node {node.node_id} from node {neighbor.node_id}")
        # if the node is not dead, remove it from the neighbor's neighbors
        if neighbor.energy > 0:
            neighbor.neighbors.pop(node.node_id)
    node.neighbors = {}


def remove_cluster_head_from_cluster(network, cluster_head):
    cluster_id = cluster_head.cluster_id
    for child in cluster_head.neighbors.values():
        if child.cluster_id == cluster_id:
            child.cluster_id = 0
            # Find the new cluster head for the child
            add_node_to_cluster(network=network, node=child)
    cluster_head.neighbors = {}


def mark_node_as_dead(node, round):
    print(f"Node {node.node_id} is dead.")
    node.round_dead = round


def save_metrics(config, name,
                 network_energy, num_dead_nodes, num_alive_nodes):
    num_nodes = config.network.num_sensor
    # Build a json object
    metrics = {
        "num_nodes": num_nodes,
        "num_rounds": config.network.protocol.rounds,
        "num_dead_nodes": num_dead_nodes,
        "num_alive_nodes": num_alive_nodes,
        "network_energy": network_energy
    }
    # Save the file as a json file
    with open(name+".json", "w") as f:
        json.dump(metrics, f)
