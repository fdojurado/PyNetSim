from gymnasium import spaces
import numpy as np
import copy

MAX_STEPS = 1000


def initialize():
    max_steps = MAX_STEPS
    action = 0
    same_cluster_heads = 0

    round_number = 0

    n_actions = 2
    action_space = spaces.Discrete(n_actions)

    # Observation are:
    # - Remaining energy of the network (float)
    # - Energy consumption of each node (array of 99)
    # - current cluster heads (array of 5)
    # - number of times the current cluster heads have been cluster heads (int)
    # - Membership of each node to a cluster (array of 99)
    # - previous action (int)
    n_obs = 1+99+5+1+99+1
    observation_space = spaces.Box(
        low=-5, high=5, shape=(n_obs,), dtype=np.float32)

    return max_steps, action, same_cluster_heads, round_number, action_space, observation_space


def copy_network(network, net_model):
    network_copy = copy.deepcopy(network)
    net_model_copy = copy.deepcopy(net_model)
    network_copy.set_model(net_model_copy)
    net_model_copy.set_network(network_copy)
    # Register callback to the network model
    # self.model.register_round_complete_callback(self.round_callback)
    # Register the callback to the network
    net_model_copy.register_round_complete_callback(
        network_copy.round_callback)
    return network_copy, net_model_copy


def get_remaining_energy(network):
    return network.remaining_energy()/50


def get_energy_consumption(network):
    energy = []
    for node in network:
        if node.node_id <= 1:
            continue
        energy.append(node.remaining_energy/0.5)
    return energy


def get_cluster_heads(network):
    cluster_heads = []
    for node in network:
        if node.node_id <= 1:
            continue
        if node.is_cluster_head:
            cluster_heads.append(node.node_id/100)
    # TODO: Fix this for the MILP case
    # sort
    if len(cluster_heads) < 5:
        cluster_heads.extend([0]*(5-len(cluster_heads)))
    cluster_heads.sort()
    return cluster_heads


def get_expected_num_cluster_heads(network, config):
    num_alive_nodes = network.alive_nodes()
    # percentage of cluster heads
    p = config.network.protocol.cluster_head_percentage
    expected_num_cluster_heads = int(num_alive_nodes * p)+1
    return expected_num_cluster_heads


def get_membership(network):
    membership = []
    for node in network:
        if node.node_id <= 1:
            continue
        membership.append(node.cluster_id/100)
    return membership


def get_obs(network, config, prev_action, same_cluster_heads):
    re = get_remaining_energy(network=network)
    assert re >= -1 and re <= 1, f"Remaining energy: {re}"
    nodes_energy_consumption = get_energy_consumption(
        network=network)
    assert len(
        nodes_energy_consumption) == 99, f"Length of nodes energy consumption: {len(nodes_energy_consumption)}"
    assert all(
        [x >= -1 and x <= 1 for x in nodes_energy_consumption]), f"Nodes energy consumption: {nodes_energy_consumption}"
    current_cluster_heads = get_cluster_heads(
        network=network)
    # expected_num_cluster_heads = get_expected_num_cluster_heads(
    #     network=network, config=config)
    # if len(current_cluster_heads) < expected_num_cluster_heads:
    #     print(
    #         f"Lenght of current cluster heads: {len(current_cluster_heads)}, expected: {expected_num_cluster_heads}")
    #     print(
    #         f"Number of alive nodes: {network.alive_nodes()}")
    #     for node in network:
    #         print(f"Node {node.node_id} energy: {node.remaining_energy}")
    #         if node.remaining_energy <= 0:
    #             print(f"Node {node.node_id} is dead")
    #     for node in network:
    #         print(
    #             f"Node {node.node_id} is cluster head: {node.is_cluster_head}")
    #     # print previous action
    #     print(f"Previous action: {prev_action}")
    # assert len(
    #     current_cluster_heads) == expected_num_cluster_heads, f"Length of current cluster heads: {len(current_cluster_heads)}, expected: {expected_num_cluster_heads}"
    assert all(
        [x >= 0 and x <= 1 for x in current_cluster_heads]), f"Current cluster heads: {current_cluster_heads}"
    num_times_ch = same_cluster_heads/400
    assert num_times_ch >= 0 and num_times_ch <= 1, f"Number of times CH: {num_times_ch}"
    membership = get_membership(network=network)
    assert len(
        membership) == 99, f"Length of membership: {len(membership)}"
    assert all(
        [x >= 0 and x <= 1 for x in membership]), f"Membership: {membership}"
    assert prev_action >= 0 and prev_action <= 1, f"Previous action: {prev_action}"
    return np.array([re, *nodes_energy_consumption, *current_cluster_heads, num_times_ch, *membership, prev_action])


def step(action, network, net_model, config, prev_action,
         same_cluster_heads, round_number, protocol, testing=False):
    action = int(action)
    done = False
    reward = 0
    stats = {}
    stats[round_number] = {}
    stats[round_number]["action"] = action

    # Two actions, create a new set of clusters or stay in the same set
    if action == 0:
        same_cluster_heads = 0
        # print("Stay in the same set of clusters")
        round_number += 1
        net_model.dissipate_energy(round=round_number)
        # reward += 0.1
    if action == 1:
        same_cluster_heads += 1
        # print("Create a new set of clusters")
        round_number = protocol.evaluate_round(
            round=round_number)
        # reward = self.episode_network.average_remaining_energy()/0.5
        reward += 0.1
    # Are there any dead nodes?
    alive_nodes = network.alive_nodes()
    if not testing:
        if alive_nodes < 99:
            done = True
            reward = 2
            print(f"Number of rounds: {round_number}")
            # print how many times the same action was taken
            action_0 = 0
            action_1 = 0
            for round in stats:
                if stats[round]["action"] == 0:
                    action_0 += 1
                else:
                    action_1 += 1
            print(f"Number of times action 0 was taken: {action_0}")
            print(f"Number of times action 1 was taken: {action_1}")
            # print the last round number
            print(f"Last round number: {round_number}")
        else:
            reward += 1
    else:
        if alive_nodes <= 0:
            done = True
            reward = 2
            print(f"Number of rounds: {round_number}")
            # print how many times the same action was taken
            action_0 = 0
            action_1 = 0
            for round in stats:
                if stats[round]["action"] == 0:
                    action_0 += 1
                else:
                    action_1 += 1
            print(f"Number of times action 0 was taken: {action_0}")
            print(f"Number of times action 1 was taken: {action_1}")
            # print the last round number
            print(f"Last round number: {round_number}")
        else:
            reward += 1

    obs = get_obs(network=network, config=config,
                  prev_action=prev_action, same_cluster_heads=same_cluster_heads)

    # print(f"Observation: {obs}")
    # input("Press enter to continue...")

    return action, same_cluster_heads, round_number, obs, reward, done, False, {}
