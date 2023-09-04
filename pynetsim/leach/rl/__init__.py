import pynetsim.leach as leach
import numpy as np


def obs(num_sensors: int, network: object,
        x_pos: np.ndarray, y_pos: np.ndarray,
        dst_to_sink: np.ndarray,
        init_energy: float,
        round: int,
        max_steps: int,
        max_distance: float,
        action_taken: int = 0):
    # Put the energy consumption in a numpy array
    energy_consumption = np.zeros(num_sensors+1)
    for node in network:
        if node.node_id == 1:
            continue
        energy = max(node.energy, 0)/init_energy
        energy_consumption[node.node_id] = energy

    # print(f"sizes: {len(energy_consumption)}, {len(network.nodes)}")

    cluster_heads = np.zeros(num_sensors+1)
    for node in network:
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
    for node in network:
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


def obs_packet_loss(num_sensors: int, network: object,
                    x_pos: np.ndarray, y_pos: np.ndarray,
                    dst_to_sink: np.ndarray,
                    init_energy: float,
                    round: int,
                    max_steps: int,
                    max_distance: float,
                    action_taken: int = 0):

    ob = obs(num_sensors, network, x_pos, y_pos, dst_to_sink,
             init_energy, round, max_steps, max_distance, action_taken)
    # Append the PDR for each node
    pdr = np.zeros(num_sensors+1)
    for node in network:
        if node.node_id == 1:
            continue
        pdr[node.node_id] = node.packet_delivery_ratio()
    ob = np.append(ob, pdr)
    # Append the network's PDR
    network_pdr = network.packet_delivery_ratio()
    ob = np.append(ob, network_pdr)
    return ob


def create_network(network: object, config: object):
    for node in network:
        network.mark_as_non_cluster_head(node)
        # use np random to set the energy
        node.energy = np.random.uniform(
            low=0.5, high=config.network.protocol.init_energy)

    # Choose 5% of the number of nodes as cluster heads
    num_cluster_heads = int(config.network.num_sensor *
                            config.network.protocol.cluster_head_percentage)

    # Choose num_cluster_heads nodes as cluster heads from the set of nodes
    # whose energy is greater or equal to the current network's average energy
    avg_energy = network.average_energy()
    # Also avoid choosing the sink as a cluster head
    cluster_heads = np.random.choice(
        [node for node in network if node.energy >= avg_energy and node.node_id != 1], size=num_cluster_heads, replace=False)

    # Set the cluster heads
    for cluster_head in cluster_heads:
        network.mark_as_cluster_head(cluster_head, cluster_head.node_id)
