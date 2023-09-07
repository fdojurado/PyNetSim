import matplotlib.pyplot as plt
import json


def add_to_metrics(config, network, round, network_energy, num_dead_nodes,
                   num_alive_nodes, num_cluster_heads, pkt_delivery_ratio,
                   pkt_loss_ratio,
                   control_packets_energy,
                   control_packet_bits,
                   pkts_sent_to_bs,
                   energy_dissipated):
    num_nodes = config.network.num_sensor
    network_energy[round] = network.remaining_energy()
    num_dead_nodes[round] = num_nodes - network.alive_nodes()
    num_alive_nodes[round] = network.alive_nodes()
    num_cluster_heads[round] = network.num_cluster_heads()
    pkt_delivery_ratio[round] = network.packet_delivery_ratio()
    pkt_loss_ratio[round] = network.average_plr()
    control_packets_energy[round] = network.control_packets_energy()
    control_packet_bits[round] = network.control_packet_bits()
    pkts_sent_to_bs[round] = network.pkts_sent_to_bs()
    energy_dissipated[round] = network.energy_dissipated()


def save_metrics(config, network_energy, num_dead_nodes, num_alive_nodes,
                 num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                 control_packets_energy, control_packet_bits,
                 pkts_sent_to_bs,
                 energy_dissipated):
    num_nodes = config.network.num_sensor
    # Build a json object
    metrics = {
        "num_nodes": num_nodes,
        "num_rounds": config.network.protocol.rounds,
        "num_dead_nodes": num_dead_nodes,
        "num_alive_nodes": num_alive_nodes,
        "network_energy": network_energy,
        "num_cluster_heads": num_cluster_heads,
        "pkt_delivery_ratio": pkt_delivery_ratio,
        "pkt_loss_ratio": pkt_loss_ratio,
        "control_packets_energy": control_packets_energy,
        "control_packet_bits": control_packet_bits,
        "pkts_sent_to_bs": pkts_sent_to_bs,
        "energy_dissipated": energy_dissipated
    }
    name = config.network.protocol.name + "_" + config.network.model
    # Save the file as a json file
    with open(name+".json", "w") as f:
        json.dump(metrics, f)

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
        elif node.is_cluster_head:
            node.color = "red"
        else:
            node.color = "blue"
        ax.plot(node.x, node.y, 'o', color=node.color)


def plot_cluster_connections(network, ax):
    cluster_heads_exist = any(
        node.is_cluster_head for node in network)
    if not cluster_heads_exist:
        # print("There are no cluster heads.")
        return
    for node in network:
        if node.is_cluster_head or node.node_id == 1:
            continue
        cluster_head = network.get_cluster_head(node=node)
        ax.plot([node.x, cluster_head.x], [
                node.y, cluster_head.y], 'k--', linewidth=0.5)

    for node in network:
        if node.is_cluster_head:
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
