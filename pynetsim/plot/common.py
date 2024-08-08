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

import os
import pandas as pd
from rich.progress import Progress

node_locations = {
    "nodes": [
        {"node_id": 1, "x": 50, "y": 175},
        {"node_id": 2, "x": 56, "y": 91},
        {"node_id": 3, "x": 21, "y": 32},
        {"node_id": 4, "x": 99, "y": 76},
        {"node_id": 5, "x": 43, "y": 27},
        {"node_id": 6, "x": 54, "y": 90},
        {"node_id": 7, "x": 70, "y": 39},
        {"node_id": 8, "x": 2, "y": 71},
        {"node_id": 9, "x": 46, "y": 46},
        {"node_id": 10, "x": 92, "y": 64},
        {"node_id": 11, "x": 94, "y": 34},
        {"node_id": 12, "x": 92, "y": 90},
        {"node_id": 13, "x": 81, "y": 12},
        {"node_id": 14, "x": 24, "y": 58},
        {"node_id": 15, "x": 51, "y": 77},
        {"node_id": 16, "x": 14, "y": 41},
        {"node_id": 17, "x": 1, "y": 57},
        {"node_id": 18, "x": 15, "y": 18},
        {"node_id": 19, "x": 64, "y": 38},
        {"node_id": 20, "x": 52, "y": 0},
        {"node_id": 21, "x": 69, "y": 31},
        {"node_id": 22, "x": 62, "y": 97},
        {"node_id": 23, "x": 10, "y": 93},
        {"node_id": 24, "x": 60, "y": 92},
        {"node_id": 25, "x": 49, "y": 95},
        {"node_id": 26, "x": 60, "y": 34},
        {"node_id": 27, "x": 61, "y": 4},
        {"node_id": 28, "x": 22, "y": 86},
        {"node_id": 29, "x": 61, "y": 79},
        {"node_id": 30, "x": 37, "y": 75},
        {"node_id": 31, "x": 42, "y": 51},
        {"node_id": 32, "x": 48, "y": 28},
        {"node_id": 33, "x": 8, "y": 52},
        {"node_id": 34, "x": 58, "y": 28},
        {"node_id": 35, "x": 3, "y": 22},
        {"node_id": 36, "x": 84, "y": 67},
        {"node_id": 37, "x": 41, "y": 59},
        {"node_id": 38, "x": 91, "y": 19},
        {"node_id": 39, "x": 25, "y": 39},
        {"node_id": 40, "x": 71, "y": 87},
        {"node_id": 41, "x": 19, "y": 99},
        {"node_id": 42, "x": 46, "y": 62},
        {"node_id": 43, "x": 9, "y": 77},
        {"node_id": 44, "x": 21, "y": 60},
        {"node_id": 45, "x": 35, "y": 59},
        {"node_id": 46, "x": 59, "y": 47},
        {"node_id": 47, "x": 66, "y": 19},
        {"node_id": 48, "x": 12, "y": 18},
        {"node_id": 49, "x": 83, "y": 30},
        {"node_id": 50, "x": 26, "y": 94},
        {"node_id": 51, "x": 89, "y": 52},
        {"node_id": 52, "x": 84, "y": 67},
        {"node_id": 53, "x": 60, "y": 85},
        {"node_id": 54, "x": 20, "y": 58},
        {"node_id": 55, "x": 77, "y": 14},
        {"node_id": 56, "x": 28, "y": 83},
        {"node_id": 57, "x": 70, "y": 30},
        {"node_id": 58, "x": 24, "y": 91},
        {"node_id": 59, "x": 50, "y": 79},
        {"node_id": 60, "x": 65, "y": 85},
        {"node_id": 61, "x": 10, "y": 29},
        {"node_id": 62, "x": 52, "y": 44},
        {"node_id": 63, "x": 53, "y": 32},
        {"node_id": 64, "x": 24, "y": 63},
        {"node_id": 65, "x": 75, "y": 0},
        {"node_id": 66, "x": 57, "y": 77},
        {"node_id": 67, "x": 80, "y": 75},
        {"node_id": 68, "x": 79, "y": 83},
        {"node_id": 69, "x": 24, "y": 62},
        {"node_id": 70, "x": 95, "y": 88},
        {"node_id": 71, "x": 21, "y": 73},
        {"node_id": 72, "x": 32, "y": 21},
        {"node_id": 73, "x": 64, "y": 15},
        {"node_id": 74, "x": 45, "y": 32},
        {"node_id": 75, "x": 24, "y": 19},
        {"node_id": 76, "x": 31, "y": 34},
        {"node_id": 77, "x": 42, "y": 94},
        {"node_id": 78, "x": 70, "y": 99},
        {"node_id": 79, "x": 98, "y": 41},
        {"node_id": 80, "x": 86, "y": 45},
        {"node_id": 81, "x": 95, "y": 29},
        {"node_id": 82, "x": 52, "y": 92},
        {"node_id": 83, "x": 12, "y": 61},
        {"node_id": 84, "x": 20, "y": 31},
        {"node_id": 85, "x": 78, "y": 43},
        {"node_id": 86, "x": 14, "y": 98},
        {"node_id": 87, "x": 90, "y": 67},
        {"node_id": 88, "x": 52, "y": 92},
        {"node_id": 89, "x": 55, "y": 0},
        {"node_id": 90, "x": 20, "y": 81},
        {"node_id": 91, "x": 75, "y": 61},
        {"node_id": 92, "x": 63, "y": 17},
        {"node_id": 93, "x": 78, "y": 70},
        {"node_id": 94, "x": 63, "y": 39},
        {"node_id": 95, "x": 34, "y": 43},
        {"node_id": 96, "x": 97, "y": 21},
        {"node_id": 97, "x": 23, "y": 35},
        {"node_id": 98, "x": 81, "y": 74},
        {"node_id": 99, "x": 82, "y": 51},
        {"node_id": 100, "x": 59, "y": 32},
    ]
}


def get_round_data(stats):
    energy_levels = list(stats['energy_levels'].values())
    # convert the energy levels to a list of integers
    energy_levels = [float(energy_level) for energy_level in energy_levels]

    membership = [0 if cluster_id is None else int(cluster_id)
                  for _, cluster_id in stats['membership'].items()]
    # Remove the sink
    membership = membership[1:]
    # Convert the membership to a list of integers
    membership = [int(cluster_id) for cluster_id in membership]

    # Get the remaining energy
    remaining_energy = stats['remaining_energy']
    # convert the remaining energy to a float
    remaining_energy = float(remaining_energy)

    # Get distance to cluster head
    dst_to_cluster_head = list(stats['dst_to_cluster_head'].values())
    # convert the distance to a list of floats
    dst_to_cluster_head = [float(dst) for dst in dst_to_cluster_head]

    # Get the alive nodes
    alive_nodes = stats['alive_nodes']
    # convert the alive nodes to an integer
    alive_nodes = int(alive_nodes)

    # Number of cluster heads
    num_cluster_heads = stats['num_cluster_heads']
    # convert the number of cluster heads to an integer
    num_cluster_heads = int(num_cluster_heads)

    # Get PDR
    pdr = stats['pdr']
    # convert the pdr to a float
    pdr = float(pdr)

    # Get the cluster heads
    if not stats['cluster_heads']:
        cluster_heads = [0] * 5
    else:
        cluster_heads = stats['cluster_heads']
        if len(cluster_heads) < 5:
            cluster_heads += [0] * (5-len(cluster_heads))

    cluster_heads.sort(reverse=False)

    # Get control packet bits
    control_packet_bits = stats['control_pkt_bits']
    # convert the control packet bits to a float
    control_packet_bits = float(control_packet_bits)

    # pkts_recv_by_bs
    pkts_recv_by_bs = stats['pkts_recv_by_bs']
    # convert the pkts_recv_by_bs to an integer
    pkts_recv_by_bs = int(pkts_recv_by_bs)

    # energy_dissipated
    energy_dissipated = stats['energy_dissipated']
    # convert the energy_dissipated to a float
    energy_dissipated = float(energy_dissipated)

    # put everything in a dictionary
    data = {
        "energy_levels": energy_levels,
        "dst_to_cluster_head": dst_to_cluster_head,
        "remaining_energy": remaining_energy,
        "alive_nodes": alive_nodes,
        "cluster_heads": cluster_heads,
        "membership": membership,
        "pdr": pdr,
        "control_pkt_bits": control_packet_bits,
        "pkts_recv_by_bs": pkts_recv_by_bs,
        "num_cluster_heads": num_cluster_heads,
        "energy_dissipated": energy_dissipated,
    }

    return data


def process_data(samples, output_folder, export_csv=True):
    # Lets create a pandas dataframe to store the data
    columns = [
        "alpha", "beta", "gamma", "remaining_energy", "alive_nodes", "cluster_heads", "energy_levels", "dst_to_cluster_head", "membership"]
    df = pd.DataFrame(columns=columns)

    # Get the size of the samples
    file_size = len(samples)

    # Initialize an empty list to store DataFrames
    dfs_list = []

    # Iterate over the samples
    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Processing samples", total=file_size)

        for name, data in samples.items():
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds - 1:
                    continue

                round_data = get_round_data(
                    stats)

                # if the name is not a number we keep it as a string
                is_number = True
                for x in name:
                    # if it is not a float, we keep it as a string
                    if not isinstance(x, float):
                        is_number = False
                        break
                if is_number:
                    name = tuple(float(x) for x in name)

                # Create a DataFrame for the current round
                df_data = pd.DataFrame({
                    "name": [name],
                    "round": [round],
                    "remaining_energy": [round_data['remaining_energy']],
                    "alive_nodes": [round_data['alive_nodes']],
                    "cluster_heads": [round_data['cluster_heads']],
                    "energy_levels": [round_data['energy_levels']],
                    "dst_to_cluster_head": [round_data['dst_to_cluster_head']],
                    "membership": [round_data['membership']],
                    "pdr": [round_data['pdr']],
                    "control_pkt_bits": [round_data['control_pkt_bits']],
                    "pkts_recv_by_bs": [round_data['pkts_recv_by_bs']],
                    "num_cluster_heads": [round_data['num_cluster_heads']],
                    "energy_dissipated": [round_data['energy_dissipated']]
                })

                # Check if the dataframe has any nan values
                if df_data.isnull().values.any():
                    raise Exception(f"Dataframe has nan values: {df_data}")

                # Append the DataFrame to the list
                dfs_list.append(df_data)

            progress.update(task, advance=1)

    # Concatenate all DataFrames in the list
    df = pd.concat(dfs_list, ignore_index=True)

    # Export the df to csv?
    if export_csv:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df.to_csv(os.path.join(output_folder, "data.csv"), index=False)

    return df
