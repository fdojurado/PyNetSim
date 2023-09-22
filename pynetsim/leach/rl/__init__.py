import pynetsim.leach as leach
import numpy as np

from sklearn.cluster import KMeans



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
        energy = max(node.remaining_energy, 0)/init_energy
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
    avg_energy = network.average_remaining_energy()

    observation = np.append(observation, avg_energy)

    # Append action taken
    observation = np.append(observation, action_taken)

    # Check that all the observations are between 0 and 1
    for ob in observation:
        assert ob >= 0 and ob <= 1, f"Observation: {ob}"

    return observation


def clustered_obs(num_sensors: int, network: object,
                  max_x: float, max_y: float,
                  init_energy: float,
                  round: int,
                  max_steps: int,
                  max_distance: float,
                  action_taken: int = 0):
    # Put the energy consumption in a numpy array
    energy_consumption = np.zeros(num_sensors)
    # Get all nodes in the network except the sink which is node_id 1
    nodes = [node for node in network if node.node_id != 1]
    for i, node in enumerate(nodes):
        energy = max(node.remaining_energy, 0)/init_energy
        energy_consumption[i] = energy
    # print(f"Energy consumption: {energy_consumption}")

    cluster_heads = np.zeros(num_sensors)
    for i, node in enumerate(nodes):
        if node.is_cluster_head:
            cluster_heads[i] = 1
    # print(f"Cluster heads: {cluster_heads}")

    observation = np.append(energy_consumption, cluster_heads)

    x_pos = np.zeros(num_sensors)
    y_pos = np.zeros(num_sensors)
    for i, node in enumerate(nodes):
        x_pos[i] = node.x/max_x
        y_pos[i] = node.y/max_y
    # print(f"X positions: {x_pos}")
    # print(f"Y positions: {y_pos}")
    observation = np.append(observation, x_pos)
    observation = np.append(observation, y_pos)

    dst_to_sink = np.zeros(num_sensors)
    for i, node in enumerate(nodes):
        dst_to_sink[i] = node.dst_to_sink/max_distance
    # print(f"Distance to sink: {dst_to_sink}")
    observation = np.append(observation, dst_to_sink)

    # append rounds
    observation = np.append(observation, round/max_steps)

    # Append network's average energy
    avg_energy = network.average_remaining_energy()/init_energy
    observation = np.append(observation, avg_energy)

    # Append action taken
    observation = np.append(observation, action_taken)

    # print(f"Lenght of observation: {len(observation)}")

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
        plr[node.node_id] = node.plr()
    observation = np.append(observation, plr)
    # Append the network's PLR
    network_plr = network.average_plr()
    observation = np.append(observation, network_plr)

    # Check that all the observations are between 0 and 1
    for ob in observation:
        assert ob >= 0 and ob <= 1, f"Observation: {ob}"

    return observation


def create_network(network: object, config: object, lower_energy: float = 0):

    # Set a random initial energy value between 50% and 100% of the initial energy
    # init_energy = np.random.uniform(
    #     low=config.network.protocol.init_energy*0.05, high=config.network.protocol.init_energy)
    init_energy = config.network.protocol.init_energy

    for node in network:
        if node.node_id == 1:
            continue
        network.mark_as_non_cluster_head(node)
        # Generate a random initial energy values with mean init_energy and standard deviation 0.1*init_energy
        # remaining_energy = np.random.normal(
        #     loc=init_energy, scale=0.01*init_energy)
        # if remaining_energy > config.network.protocol.init_energy:
        #     remaining_energy = config.network.protocol.init_energy
        node.remaining_energy = max(init_energy, 0)
        # packet sent and received are set to 0 by default
        node.round_dead = 0
        node.clear_stats()


def create_clustered_network(network: object, config: object, lower_energy: float = 0):

    create_network(network, config, lower_energy)

    # Number of clusters
    num_clusters = np.ceil(
        network.alive_nodes() * config.network.protocol.cluster_head_percentage)
    # print(f"Number of clusters: {num_clusters}")
    # print the number of alive nodes
    # print(f"Number of alive nodes: {network.alive_nodes()}")
    #  x and y coordinates of nodes
    x = []
    y = []
    for node in network:
        if network.should_skip_node(node):
            continue
        x.append(node.x)
        y.append(node.y)
    coordinates = np.array(list(zip(x, y)))
    # print(f"Coordinates: {coordinates}")
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10)
    # fit the model to the coordinates
    kmeans.fit(coordinates)
    # get the cluster centers
    centers = kmeans.cluster_centers_
    # get the cluster labels
    labels = kmeans.labels_
    # Assign cluster ids to the nodes
    for node in network:
        if network.should_skip_node(node):
            continue
        # get index of node in coordinates
        index = np.where((coordinates[:, 0] == node.x) & (
            coordinates[:, 1] == node.y))
        # print(f"Node: {node.node_id}, index: {index}")
        # get label of node
        label = labels[index[0][0]]
        # print(f"Node: {node.node_id}, label: {label}")
        node.cluster_id = label
    # Assign cluster heads
    for cluster_id in range(int(num_clusters)):
        cluster_nodes = []
        for node in network:
            if network.should_skip_node(node):
                continue
            if node.cluster_id == cluster_id:
                cluster_nodes.append(node)
        # get the cluster head with the highest remaining energy
        cluster_head = max(
            cluster_nodes, key=lambda node: node.remaining_energy)
        # print(f"Node {cluster_head.node_id} is cluster head with cluster id {cluster_head.cluster_id}")
        # set the cluster head
        network.mark_as_cluster_head(
            cluster_head, cluster_head.cluster_id)
    # Assign the distance to cluster head
    for node in network:
        if network.should_skip_node(node):
            continue
        if node.is_cluster_head:
            node.dst_to_cluster_head = node.dst_to_sink
        else:
            cluster_head = network.get_cluster_head(node)
            # print(f"Node {cluster_head.node_id} is cluster head with cluster id {cluster_head.cluster_id} of node {node.node_id}")
            node.dst_to_cluster_head = network.distance_between_nodes(
                node, cluster_head)
