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
    avg_energy = network.average_energy()/init_energy

    observation = np.append(observation, avg_energy)

    # Append action taken
    observation = np.append(observation, action_taken)

    # Check that all the observations are between 0 and 1
    for ob in observation:
        assert ob >= 0 and ob <= 1, f"Observation: {ob}"

    return observation


def obs_packet_loss(num_sensors: int, network: object,
                    x_pos: np.ndarray, y_pos: np.ndarray,
                    dst_to_sink: np.ndarray,
                    init_energy: float,
                    round: int,
                    max_steps: int,
                    max_distance: float,
                    action_taken: int = 0):

    observation = obs(num_sensors, network, x_pos, y_pos, dst_to_sink,
                      init_energy, round, max_steps, max_distance, action_taken)
    # Append the PLR for each node
    plr = np.zeros(num_sensors+1)
    for node in network:
        if node.node_id == 1:
            continue
        plr[node.node_id] = node.packet_loss_ratio()
    observation = np.append(observation, plr)
    # Append the network's PLR
    network_plr = network.average_plr()
    observation = np.append(observation, network_plr)

    # Check that all the observations are between 0 and 1
    for ob in observation:
        assert ob >= 0 and ob <= 1, f"Observation: {ob}"

    return observation


def create_network(network: object, config: object):
    for node in network:
        network.mark_as_non_cluster_head(node)
        # use np random to set the energy
        node.energy = np.random.uniform(
            low=0, high=config.network.protocol.init_energy)
        # packet sent and received are set to 0 by default
        node.round_dead = 0
        node.packet_sent = 0
        node.packet_received = 0
        node.clear_control_packet_bits()
        node.clear_energy_control_packet()
        # node.energy_control_packets = 0

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
