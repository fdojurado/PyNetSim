#     PyNetSim: A Python-based Network Simulator for Low-Energy Adaptive Clustering Hierarchy (LEACH) Protocol
#     Copyright (C) 2024  F. Fernando Jurado-Lasso (ffjla@dtu.dk)

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pynetsim.plot import common


TICK_SIZE = 23
LABEL_SIZE = 24
LEGEND_SIZE = 25
LINE_WIDTH = 3.5

protocols = {
    'leach-rlc': {
        'name': 'LEACH-RLC',
        'color': 'blue',
        'line_style': '-',
        'hatch_style': ''
    },
    'leach-rle': {
        'name': 'LEACH-RLE',
        'color': 'black',
        'line_style': '-',
        'hatch_style': ''
    },
    'leach': {
        'name': 'LEACH',
        'color': 'orange',
        'line_style': 'dashed',
        'hatch_style': ''
    },
    'leach-c': {
        'name': 'LEACH-C',
        'color': 'green',
        'line_style': 'dashed',
        'hatch_style': ''
    },
    'ee-leach': {
        'name': 'EE-LEACH',
        'color': 'red',
        'line_style': 'dashed',
        'hatch_style': ''
    },
    'leach-d': {
        'name': 'LEACH-D',
        'color': 'purple',
        'line_style': 'dashed',
        'hatch_style': ''
    },
    'leach-cm': {
        'name': 'LEACH-CM',
        'color': 'brown',
        'line_style': 'dashed',
        'hatch_style': ''
    }
}


def get_name(name):
    name = name.split('/')[-2]
    return name


def calculate_mean_and_ci(data):
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    ci = 1.96 * (std / np.sqrt(n))
    return mean, ci


def process_data(df):
    grouped = df.groupby('round')
    results = {}

    for name, group in grouped:
        mean_remaining_energy, ci_remaining_energy = calculate_mean_and_ci(
            group['remaining_energy'])
        # mean_dead_nodes, ci_dead_nodes = calculate_mean_and_ci(
        #     group['dead_nodes'])
        mean_alive_nodes, ci_alive_nodes = calculate_mean_and_ci(
            group['alive_nodes'])
        mean_pdr, ci_pdr = calculate_mean_and_ci(group['pdr'])
        # mean_plr, ci_plr = calculate_mean_and_ci(group['plr'])
        # mean_control_packets_energy, ci_control_packets_energy = calculate_mean_and_ci(
        #     group['control_packets_energy'])
        mean_control_pkt_bits, ci_control_pkt_bits = calculate_mean_and_ci(
            group['control_pkt_bits'])
        # mean_pkts_sent_to_bs, ci_pkts_sent_to_bs = calculate_mean_and_ci(
        #     group['pkts_sent_to_bs'])
        mean_energy_dissipated, ci_energy_dissipated = calculate_mean_and_ci(
            group['energy_dissipated'])
        mean_pkts_recv_by_bs, ci_pkts_recv_by_bs = calculate_mean_and_ci(
            group['pkts_recv_by_bs'])
        results[name] = {
            "mean_remaining_energy": mean_remaining_energy,
            "ci_remaining_energy": ci_remaining_energy,
            # "mean_dead_nodes": mean_dead_nodes,
            # "ci_dead_nodes": ci_dead_nodes,
            "mean_alive_nodes": mean_alive_nodes,
            "ci_alive_nodes": ci_alive_nodes,
            "mean_pdr": mean_pdr,
            "ci_pdr": ci_pdr,
            # "mean_plr": mean_plr,
            # "ci_plr": ci_plr,
            # "mean_control_packets_energy": mean_control_packets_energy,
            # "ci_control_packets_energy": ci_control_packets_energy,
            "mean_control_pkt_bits": mean_control_pkt_bits,
            "ci_control_pkt_bits": ci_control_pkt_bits,
            # "mean_pkts_sent_to_bs": mean_pkts_sent_to_bs,
            # "ci_pkts_sent_to_bs": ci_pkts_sent_to_bs,
            "mean_energy_dissipated": mean_energy_dissipated,
            "ci_energy_dissipated": ci_energy_dissipated,
            "mean_pkts_recv_by_bs": mean_pkts_recv_by_bs,
            "ci_pkts_recv_by_bs": ci_pkts_recv_by_bs,
        }

    grouped = df.groupby('name')
    ch_count = {}
    for name, group in grouped:
        cluster_heads = group['cluster_heads']

        cluster_heads_flat = [
            item for sublist in cluster_heads for item in sublist]
        cluster_heads_count = pd.Series(cluster_heads_flat).value_counts()
        ch_count[name] = cluster_heads_count

    ch_count_df = pd.DataFrame(ch_count)
    ch_count_mean = ch_count_df.mean(axis=1)

    ch_count = {}
    for i in range(101):
        if i == 1:
            continue
        ch_count[i] = ch_count_mean[i]

    # convert the results to a DataFrame
    results = pd.DataFrame(results).T

    # -------- #
    grouped = df.groupby('name')
    ch_count_two = {}
    for name, group in grouped:
        num_cluster_heads = group['num_cluster_heads']
        num_cluster_heads = num_cluster_heads.drop(num_cluster_heads.index[0])
        # Count the frequency of the number of cluster heads
        ch_count_two[name] = num_cluster_heads.value_counts()

    ch_count_two = pd.DataFrame(ch_count_two)
    ch_count_two = ch_count_two.fillna(0)
    ch_count_two = ch_count_two.T
    ch_count_mean = ch_count_two.mean()
    ch_count_ci = ch_count_two.sem() * 1.96
    ch_count_two = pd.DataFrame({
        "mean": ch_count_mean,
        "ci": ch_count_ci
    })

    return results, ch_count, ch_count_two


def load_input_data(input_files):
    results = {}
    for input_file in os.listdir(input_files):
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        with open(os.path.join(input_files, input_file), "r") as f:
            data = json.load(f)
            results[file_name] = data
    return results


def plot_alive_nodes_vs_rounds(results, output_folder):
    scaling_factor = 1
    fig, ax = plt.subplots(figsize=(10, 7))
    x_max = 0

    for key, value in results.items():
        value = value['processed_data']
        mean_alive_nodes = value['mean_alive_nodes']
        ci_alive_nodes = value['ci_alive_nodes']
        max_rounds = len(mean_alive_nodes)
        rounds = list(mean_alive_nodes.keys())
        proto = protocols[get_name(key)]
        ax.plot(rounds, mean_alive_nodes, label=proto['name'],
                linewidth=LINE_WIDTH, color=proto['color'], linestyle=proto['line_style'])
        ax.fill_between(rounds, mean_alive_nodes - ci_alive_nodes, mean_alive_nodes + ci_alive_nodes,
                        alpha=0.2)
        if max_rounds > x_max:
            x_max = max_rounds

    ax.set_xlim(700, int(x_max) + 1)
    ax.tick_params(axis='both', which='major',
                   labelsize=TICK_SIZE * scaling_factor)
    ax.set_xlabel("Round", fontsize=LABEL_SIZE * scaling_factor)
    ax.set_ylabel("Alive Nodes", fontsize=LABEL_SIZE * scaling_factor)
    ax.legend(fontsize=LEGEND_SIZE * scaling_factor)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, "alive_nodes_vs_rounds.pdf"))
    plt.close()


def plot_first_dead_round(results, output_folder):
    # Get the values and the names
    values = []
    names = []
    error = []
    for key, value in results.items():
        value = value['processed_data']
        mean_alive_nodes = value['mean_alive_nodes']
        ci_alive_nodes = value['ci_alive_nodes']
        dead_round = [round for round,
                      alive_nodes in mean_alive_nodes.items() if alive_nodes < 99]
        values.append(dead_round[0])
        error.append(ci_alive_nodes[dead_round[0]])
        names.append(get_name(key))
    # Sort the values in descending order
    # Sort the values in ascending order for plotting
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    sorted_values = [values[i] for i in sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_error = [error[i] for i in sorted_indices]

    scaling_factor = 1
    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect handles and labels in the order of sorted_names
    handles = []
    labels = []

    for i, name in enumerate(sorted_names):
        label = protocols[name]['name']
        color = protocols[name]['color']
        bar = ax.bar(label, sorted_values[i], color=color, edgecolor='black',
                     hatch=protocols[name]['hatch_style'], linewidth=2.5, zorder=2)
        # if label == 'LEACH-RLC':
        #     bar = ax.bar(
        #         label, sorted_values[i], color=color, label=label, edgecolor='black', zorder=2, hatch=protocols[name]['hatch_style'], linewidth=2.5)
        # else:
        #     bar = ax.bar(
        #         label, sorted_values[i], color='none', label=label, edgecolor=color, zorder=2, hatch=protocols[name]['hatch_style'], linewidth=2.5)
        ax.errorbar(x=label, y=sorted_values[i], yerr=sorted_error[i],
                    capsize=7, color='red', fmt='none', zorder=3, elinewidth=4, capthick=4)
        ax.grid(axis='y', linestyle='--', linewidth=2, zorder=1)
        handles.append(bar)
        labels.append(label)

    # Set x-ticks to the bar positions but remove labels
    ax.set_xticks(range(len(sorted_names)))
    # Replace x-tick labels with empty strings
    ax.set_xticklabels([''] * len(sorted_names))

    ax.tick_params(axis='y', which='major',
                   labelsize=TICK_SIZE * scaling_factor)
    ax.set_ylim(700, 950)
    ax.set_ylabel("FND", fontsize=LABEL_SIZE * scaling_factor)

    # Create a dictionary to map labels to values for sorting the legend
    label_value_map = {protocols[name]['name']: sorted_values[i]
                       for i, name in enumerate(sorted_names)}

    # Sort handles and labels according to values in descending order for the legend
    sorted_handles_labels = sorted(
        zip(handles, labels), key=lambda t: label_value_map[t[1]], reverse=True)
    sorted_handles, sorted_labels = zip(
        *sorted_handles_labels) if sorted_handles_labels else ([], [])

    # Set the legend
    ax.legend(sorted_handles, sorted_labels,
              fontsize=LEGEND_SIZE * scaling_factor)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, "first_dead_rounds.pdf"))
    plt.close()


def plot_pdr_vs_rounds(results, output_folder):
    scaling_factor = 1
    fig, ax = plt.subplots(figsize=(10, 7))
    x_max = 0

    for key, value in results.items():
        value = value['processed_data']
        mean_pdr = value['mean_pdr']
        ci_pdr = value['ci_pdr']
        max_rounds = len(mean_pdr)
        rounds = list(mean_pdr.keys())
        # Ensure this function is defined elsewhere
        name = protocols[get_name(key)]['name']
        ax.plot(rounds, mean_pdr, label=name, linewidth=LINE_WIDTH,
                color=protocols[get_name(key)]['color'], linestyle=protocols[get_name(key)]['line_style'])
        ax.fill_between(rounds, mean_pdr - ci_pdr, mean_pdr + ci_pdr,
                        alpha=0.2)

        if max_rounds > x_max:
            x_max = max_rounds

    ax.set_xlim(0, int(x_max) + 1)
    ax.set_ylim(0.65, 1.01)
    ax.set_xlabel("Round", fontsize=LABEL_SIZE * scaling_factor)
    ax.set_ylabel("PDR", fontsize=LABEL_SIZE * scaling_factor)
    ax.tick_params(axis='both', which='major',
                   labelsize=TICK_SIZE * scaling_factor)

    # Set legend
    ax.legend(fontsize=LEGEND_SIZE * scaling_factor)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, "pdr_vs_rounds.pdf"))
    plt.close()


def plot_average_energy_dissipated(results, output_folder):
    scaling_factor = 1
    fig, ax = plt.subplots(figsize=(10, 7))
    x_max = 0

    for key, value in results.items():
        value = value['processed_data']
        remaining_energy = value['mean_remaining_energy']
        ci_remaining_energy = value['ci_remaining_energy']
        mean_alive_nodes = value['mean_alive_nodes']

        # Calculate average energy dissipated
        average_energy_dissipated = -1 * remaining_energy.diff() / mean_alive_nodes
        average_energy_dissipated = average_energy_dissipated.drop(
            average_energy_dissipated.index[0])

        max_rounds = len(remaining_energy)
        rounds = list(average_energy_dissipated.keys())

        name = protocols[get_name(key)]['name']
        color = protocols[get_name(key)]['color']
        # Use dotted line for all except 'leach-rlc'
        linestyle = 'dotted' if name != 'LEACH-RLC' else '-'

        ax.plot(rounds, average_energy_dissipated, label=name,
                linewidth=LINE_WIDTH, color=color, linestyle=linestyle)

        ci_remaining_energy = ci_remaining_energy.drop(
            ci_remaining_energy.index[0])
        # Uncomment if you want to include the confidence interval
        # ax.fill_between(rounds, average_energy_dissipated - ci_remaining_energy,
        #                 average_energy_dissipated + ci_remaining_energy, alpha=0.2, color=color)

        if max_rounds > x_max:
            x_max = max_rounds

    ax.set_xlim(0, int(x_max) + 1)
    ax.set_ylim(0, 0.002)
    ax.tick_params(axis='both', which='major',
                   labelsize=TICK_SIZE * scaling_factor)
    ax.set_xlabel("Round", fontsize=LABEL_SIZE * scaling_factor)
    ax.set_ylabel(
        "Average Energy Dissipated [J]", fontsize=LABEL_SIZE * scaling_factor)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

    # Set legend
    ax.legend(fontsize=LEGEND_SIZE * scaling_factor)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, "average_energy_dissipated.pdf"))
    plt.close()


def plot_remaining_energy_vs_rounds(results, output_folder):
    scaling_factor = 1
    fig, ax = plt.subplots(figsize=(10, 7))
    x_max = 0

    for key, value in results.items():
        value = value['processed_data']
        remaining_energy = value['mean_remaining_energy']
        ci_remaining_energy = value['ci_remaining_energy']
        max_rounds = len(remaining_energy)
        rounds = list(remaining_energy.keys())

        name = protocols[get_name(key)]['name']
        color = protocols[get_name(key)]['color']

        ax.plot(rounds, remaining_energy, label=name,
                linewidth=LINE_WIDTH / 2, color=color, linestyle=protocols[get_name(key)]['line_style'])
        ax.fill_between(rounds, remaining_energy - ci_remaining_energy,
                        remaining_energy + ci_remaining_energy, alpha=0.2, color=color)

        if max_rounds > x_max:
            x_max = max_rounds

    ax.set_xlim(0, 1200)
    ax.set_xlabel("Round", fontsize=LABEL_SIZE * scaling_factor)
    ax.set_ylabel("Remaining Energy [J]", fontsize=LABEL_SIZE * scaling_factor)
    ax.tick_params(axis='both', which='major',
                   labelsize=TICK_SIZE * scaling_factor)

    # Set legend
    ax.legend(fontsize=LEGEND_SIZE * scaling_factor)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, "remaining_energy_vs_rounds.pdf"))
    plt.close()


def plot_control_pkt_bits_vs_rounds(results, output_folder):
    scaling_factor = 1
    fig, ax = plt.subplots(figsize=(10, 7))
    x_max = 0

    for key, value in results.items():
        value = value['processed_data']
        mean_control_pkt_bits = value['mean_control_pkt_bits']
        ci_control_pkt_bits = value['ci_control_pkt_bits']
        max_rounds = len(mean_control_pkt_bits)
        rounds = list(mean_control_pkt_bits.keys())

        name = protocols[get_name(key)]['name']
        color = protocols[get_name(key)]['color']

        ax.plot(rounds, mean_control_pkt_bits, label=name,
                linewidth=LINE_WIDTH, color=color, linestyle=protocols[get_name(key)]['line_style'])
        ax.fill_between(rounds, mean_control_pkt_bits - ci_control_pkt_bits,
                        mean_control_pkt_bits + ci_control_pkt_bits, alpha=0.2, color=color)

        if max_rounds > x_max:
            x_max = max_rounds

    ax.set_xlim(0, int(x_max) + 1)
    ax.set_xlabel("Round", fontsize=LABEL_SIZE * scaling_factor)
    ax.set_ylabel("Control Packet Bits", fontsize=LABEL_SIZE * scaling_factor)
    ax.tick_params(axis='both', which='major',
                   labelsize=TICK_SIZE * scaling_factor)

    # Set legend
    ax.legend(fontsize=LEGEND_SIZE * scaling_factor)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, "control_pkt_bits_vs_rounds.pdf"))
    plt.close()


def plot_cluster_heads_heatmap(results, output_folder):
    scaling_factor = 1.5
    for key_file, value in results.items():
        name = key_file.split("/")[-2]
        x_loc = []
        y_loc = []
        count = []

        ch_counts = value['ch_count']
        for key, value in ch_counts.items():
            desired_node_id = int(key)
            if desired_node_id <= 1:
                continue
            # Find the node with the specified node_id
            desired_node = next(
                (node for node in common.node_locations["nodes"] if node["node_id"] == desired_node_id), None)

            # Check if the node is found
            if desired_node:
                x_location = desired_node["x"]
                y_location = desired_node["y"]
            else:
                raise ValueError(
                    f"Node with node_id {desired_node_id} not found.")
            x_loc.append(x_location)
            y_loc.append(y_location)
            count.append(value)

        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(x_loc, y_loc, c=count, cmap='viridis_r',
                        s=100)  # Adjust 's' for marker size

        if name == 'leach-rl':
            cbar = plt.colorbar(sc, ax=ax)
            cbar.ax.tick_params(labelsize=TICK_SIZE * scaling_factor)
        # plt.colorbar(sc, ax=ax, label='Count')  # Add color bar

        ax.tick_params(axis='both', which='major',
                       labelsize=TICK_SIZE * scaling_factor)
        ax.set_xlabel('X Location', fontsize=LABEL_SIZE * scaling_factor)
        ax.set_ylabel('Y Location', fontsize=LABEL_SIZE * scaling_factor)
        ax.set_title(f'{protocols[name]["name"]}',
                     fontsize=LABEL_SIZE * scaling_factor)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,
                    f"cluster_heads_heatmap_{name}.pdf"))
        plt.close()


def plot_num_cluster_heads(results, output_folder):
    scaling_factor = 3

    for key, value in results.items():
        value = value['ch_count_two']
        mean = value['mean']
        ci = value['ci']
        max_rounds = len(mean)
        rounds = list(mean.keys())
        name = protocols[get_name(key)]['name']
        color = protocols[get_name(key)]['color']
        # bar plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(rounds, mean, label=name,
               color=color, edgecolor='black', hatch=protocols[get_name(key)]['hatch_style'], linewidth=2.5, zorder=2)
        ax.errorbar(rounds, mean, yerr=ci, capsize=10, elinewidth=4, capthick=4,
                    fmt='none', color='red', zorder=3)
        ax.tick_params(axis='both', which='major',
                       labelsize=TICK_SIZE * scaling_factor)
        ax.set_xlabel(r'$|CH|$', fontsize=LABEL_SIZE * scaling_factor)
        # grid
        ax.grid(axis='y', linestyle='--', linewidth=3, zorder=1)
        # minor grid
        ax.minorticks_on()
        ax.grid(axis='y', which='minor',
                linestyle=':', linewidth=2.5, zorder=1)
        ax.set_ylabel("Frequency", fontsize=LABEL_SIZE * scaling_factor)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE*scaling_factor)
        ax.set_ylim(0, 1100)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,
                                 f"{name}_num_cluster_heads_histogram.pdf"))
        plt.close()


def plot_num_cluster_heads_histogram(results, output_folder):
    scaling_factor = 3

    for key_file, value in results.items():
        df = value['df']
        grouped = df.groupby('name')
        for name, group in grouped:
            if name not in ['rl_0', 'leach_0', 'leach-c_0', 'ee-leach_0', 'leach-d_0', 'leach-cm_0']:
                continue
            if name == 'rl_0':
                name = 'leach-rl'
            elif name == 'leach_0':
                name = 'leach'
            elif name == 'leach-c_0':
                name = 'leach-c'
            elif name == 'ee-leach_0':
                name = 'ee-leach'
            elif name == 'leach-d_0':
                name = 'leach-d'
            elif name == 'leach-cm_0':
                name = 'leach-cm'
            proto = protocols[name]
            ch_count = group['num_cluster_heads']
            # Lets drop the first value since it is always 0
            ch_count = ch_count.drop(ch_count.index[0])
            y = list(map(int, ch_count.values))
            ch_count = ch_count[ch_count > 0]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(y, bins=10, alpha=0.65, label=proto['name'],
                    edgecolor='k')
            ax.tick_params(axis='both', which='major',
                           labelsize=TICK_SIZE * scaling_factor)
            ax.set_xlabel(r'$|CH|$', fontsize=LABEL_SIZE * scaling_factor)
            ax.set_ylabel("Frequency", fontsize=LABEL_SIZE * scaling_factor)
            #     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # limit the ya axis to range from 0 to 1000 fixed
            ax.set_ylim(0, 1100)
            # fix the x axis to range from 0 to 15 fixed
            # ax.set_xlim(0, 15)
            ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE*scaling_factor)
            # title
            # ax.set_title(f'{proto["name"]}', fontsize=LABEL_SIZE * scaling_factor)
            # tight layout
            plt.tight_layout()
            # save
            plt.savefig(os.path.join(output_folder,
                        f"{proto['name']}_num_cluster_heads_histogram.pdf"))
            plt.close()


def plot_results(results, output_folder):
    plot_alive_nodes_vs_rounds(results, output_folder)
    plot_first_dead_round(results, output_folder)
    plot_pdr_vs_rounds(results, output_folder)
    plot_average_energy_dissipated(results, output_folder)
    plot_remaining_energy_vs_rounds(results, output_folder)
    plot_control_pkt_bits_vs_rounds(results, output_folder)
    plot_cluster_heads_heatmap(results, output_folder)
    plot_num_cluster_heads(results, output_folder)
    # plot_num_cluster_heads_histogram(results, output_folder)


def process_results(files, output):
    dfs = {}
    for input_file in files:
        input_files = load_input_data(input_file)
        df = common.process_data(input_files, output, export_csv=False)
        processed_data, ch_count, ch_count_two = process_data(df)
        dfs[input_file] = {
            "processed_data": processed_data, "ch_count": ch_count, "ch_count_two": ch_count_two,
            "df": df}
    return dfs
