import pynetsim.leach as leach
import numpy as np


def add_rm_obs(num_sensors: int, network: object):
    # Put the energy consumption in a numpy array
    energy_consumption = np.zeros(num_sensors+1)
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        energy = max(node.energy, 0)/2
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
    x_locations = np.zeros(num_sensors+1)
    y_locations = np.zeros(num_sensors+1)
    for node in network.nodes.values():
        if node.node_id == 1:
            continue
        x_locations[node.node_id] = node.x
        y_locations[node.node_id] = node.y

    observation = np.append(observation, x_locations)
    observation = np.append(observation, y_locations)

    # append rounds
    # observation = np.append(observation, self.round)
    info = {}

    return observation, info


def dissipate_energy(round: int, network: object,
                     elect: float, eda: float, packet_size: int, eamp: float):
    leach.energy_dissipation_non_cluster_heads(round=round, network=network,
                                               elect=elect, eda=eda,
                                               packet_size=packet_size, eamp=eamp)
    leach.energy_dissipation_cluster_heads(round=round, network=network,
                                           elect=elect, eda=eda,
                                           packet_size=packet_size, eamp=eamp)
