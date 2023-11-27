import json
import os
import numpy as np
import pandas as pd

from rich.progress import Progress


def get_mean_std_name(data):
    name_values = data['name']
    name_values = name_values.apply(lambda x: eval(x))
    alpha_values = name_values.apply(lambda x: x[0])
    beta_values = name_values.apply(lambda x: x[1])
    gamma_values = name_values.apply(lambda x: x[2])
    weight_values = {
        'alpha': {
            'mean': alpha_values.mean(),
            'std': alpha_values.std()
        },
        'beta': {
            'mean': beta_values.mean(),
            'std': beta_values.std()
        },
        'gamma': {
            'mean': gamma_values.mean(),
            'std': gamma_values.std()
        }
    }
    return weight_values


def compute_stats(data):
    # data = data.apply(lambda x: eval(x))
    data_mean = data.mean()
    data_std = data.std()
    data_stats_dict = {
        'mean': data_mean,
        'std': data_std
    }
    return data_stats_dict


def compute_array_stats(data):
    data = data.apply(lambda x: eval(x))
    data_mean = data.apply(lambda x: np.mean(x)).mean()
    data_std = data.apply(lambda x: np.std(x)).mean()
    data_stats_dict = {
        'mean': data_mean,
        'std': data_std
    }
    return data_stats_dict

# Standardize data using F1 score


def standardize_inputs(x, mean, std):
    standardized_x = (x - mean) / std
    return standardized_x


def get_estimate_tx_energy(network, eelect, eamp, efs, eda, packet_size, d0):
    tx_energy = {}
    for node in network:
        if node.node_id == 1:
            continue
        tx_energy[node.node_id] = {}
        for other_node in network:
            # avoid calculating the distance between a node and itself
            if node.node_id == other_node.node_id:
                tx_energy[node.node_id][other_node.node_id] = 0
                continue
            dst = network.distance_between_nodes(node, other_node)
            eamp_calc = 0
            if dst <= d0:
                eamp_calc = packet_size*efs*dst**2
            else:
                eamp_calc = packet_size*eamp*dst**4
            if other_node.node_id == 1:
                tx_energy[node.node_id][other_node.node_id] = (
                    eelect + eda) * packet_size + eamp_calc
            else:
                tx_energy[node.node_id][other_node.node_id] = eelect * \
                    packet_size + eamp_calc
    return tx_energy


def get_standardized_weights(alpha_val, beta_val, gamma_val, data_stats):
    # Get the mean and std of the data
    alpha = standardize_inputs(
        x=alpha_val,
        mean=data_stats.loc[data_stats['name']
                            == 'alpha']['mean'].values[0],
        std=data_stats.loc[data_stats['name'] == 'alpha']['std'].values[0])
    beta = standardize_inputs(
        x=beta_val,
        mean=data_stats.loc[data_stats['name']
                            == 'beta']['mean'].values[0],
        std=data_stats.loc[data_stats['name'] == 'beta']['std'].values[0])
    gamma = standardize_inputs(
        x=gamma_val,
        mean=data_stats.loc[data_stats['name']
                            == 'gamma']['mean'].values[0],
        std=data_stats.loc[data_stats['name'] == 'gamma']['std'].values[0])
    return alpha, beta, gamma


def get_standardized_energy_levels(network: object, mean: float, std: float):
    # Get the energy levels
    np_energy_levels = np.zeros(99)
    for node in network:
        if node.node_id <= 1:
            continue
        np_energy_levels[node.node_id-2] = node.remaining_energy
    energy_levels = list(np_energy_levels)
    # print(f"Energy levels: {energy_levels}")
    return standardize_inputs(
        x=energy_levels,
        mean=mean,
        std=std
    )
