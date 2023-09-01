import pynetsim.leach as leach
import numpy as np


def add_rm_obs(num_sensors: int, network: object,
               x_pos: np.ndarray, y_pos: np.ndarray,
               dst_to_sink: np.ndarray,
               init_energy: float,
               round: int,
               max_steps: int,
               max_distance: float,
               action_taken: int = 0):
    # Put the energy consumption in a numpy array
    energy_consumption = np.zeros(num_sensors+1)
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        energy = max(node.energy, 0)/init_energy
        energy_consumption[node.node_id] = energy

    # print(f"sizes: {len(energy_consumption)}, {len(network.nodes)}")

    cluster_heads = np.zeros(num_sensors+1)
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        if node.is_cluster_head:
            cluster_heads[node.node_id] = 1

    # print(f"Size of cluster heads: {len(cluster_heads)}")

    observation = np.append(energy_consumption, cluster_heads)
    # Append the sensor nodes location
    observation = np.append(observation, x_pos)
    observation = np.append(observation, y_pos)

    # append distance to sink
    observation = np.append(observation, dst_to_sink)

    # append rounds
    observation = np.append(observation, round/max_steps)

    # Append all sensor nodes' distance to cluster head. In the case,
    # of cluster heads, append distance to sink
    dst_to_cluster_head = np.zeros(num_sensors+1)
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        if node.is_cluster_head:
            dst_to_cluster_head[node.node_id] = node.dst_to_sink/max_distance
        else:
            dst_to_cluster_head[node.node_id] = node.dst_to_cluster_head/max_distance

    observation = np.append(observation, dst_to_cluster_head)

    # Append average energy of the network
    avg_energy = network.average_energy()

    observation = np.append(observation, avg_energy)

    # Append action taken
    observation = np.append(observation, action_taken)

    return observation


def dissipate_energy(round: int, network: object,
                     elect: float, eda: float, packet_size: int, eamp: float):
    leach.energy_dissipation_non_cluster_heads(round=round, network=network,
                                               elect=elect, eda=eda,
                                               packet_size=packet_size, eamp=eamp)
    leach.energy_dissipation_cluster_heads(round=round, network=network,
                                           elect=elect, eda=eda,
                                           packet_size=packet_size, eamp=eamp)


def plot_clusters(network: object, round: int, ax: object):
    leach.plot_clusters(network=network, round=round, ax=ax)


def store_metrics(config, network, round, network_energy, num_dead_nodes, num_alive_nodes):
    leach.store_metrics(config, network, round, network_energy,
                        num_dead_nodes, num_alive_nodes)


def save_metrics(config, name, network_energy, num_dead_nodes, num_alive_nodes):
    leach.save_metrics(config, name, network_energy,
                       num_dead_nodes, num_alive_nodes)


def create_clusters(network: object):
    leach.create_clusters(network=network)


def mark_as_cluster_head(node):
    node.is_cluster_head = True
    node.cluster_id = node.node_id


def mark_as_non_cluster_head(node):
    node.is_cluster_head = False
    node.cluster_id = 0


def get_energy_conversion_factors(config):
    return leach.get_energy_conversion_factors(config)
