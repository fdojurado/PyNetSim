import copy


def copy_network(network: object, net_model: object):
    network_copy = copy.deepcopy(network)
    net_model_copy = copy.deepcopy(net_model)
    net_model_copy.set_network(network_copy)
    return network_copy, net_model_copy


def dist_between_nodes(network: object, node1: int, node2: int):
    node1 = network.get_node(node1)
    node2 = network.get_node(node2)
    return network.distance_between_nodes(node1, node2)


def energy_spent_non_ch(network: object, src: int, dst: int):
    src = network.get_node(src)
    dst = network.get_node(dst)
    return network.calculate_energy_tx_non_ch(
        dist_between_nodes(network, src.node_id, dst.node_id))


def energy_spent_ch(network: object, src: int):
    src = network.get_node(src)
    # print(f"CH: {src.node_id}, dst_to_sink: {src.dst_to_sink}")
    return network.calculate_energy_tx_ch(src.dst_to_sink)


def update_cluster_heads(network: object, chs: list):
    for node in network:
        if node.node_id in chs:
            network.mark_as_cluster_head(
                node, node.node_id)


def update_chs_to_nodes(network: object, assignments: dict):
    for node in assignments:
        src = network.get_node(node)
        ch = network.get_node(assignments[node])
        src.dst_to_cluster_head = dist_between_nodes(
            network, src.node_id, ch.node_id)
        src.cluster_id = ch.node_id


def get_energy(node: object):
    return node.remaining_energy
