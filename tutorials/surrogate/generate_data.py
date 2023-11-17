# Generate the data from JSON files.
import os
import json
import pandas as pd
import argparse
from rich.progress import Progress
from pynetsim.config import load_config
from pynetsim.network.network import Network
from pynetsim.network.extended_model import Extended

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "surrogate.yml")


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


def process_data(samples, output_folder, network, export_csv=True):
    # Lets create a pandas dataframe to store the data
    columns = [
        "alpha", "beta", "gamma", "remaining_energy", "alive_nodes", "cluster_heads", "energy_levels", "dst_to_cluster_head", "membership",
        "eelect", "pkt_size", "eamp", "efs", "eda", "d0"]
    df = pd.DataFrame(columns=columns)

    # Get the size of the samples
    file_size = len(samples)

    # Initialize an empty list to store DataFrames
    dfs_list = []
    d0 = (10 * 10**(-12) / (0.0013 * 10**(-12)))**0.5

    # Calculate the minimum, maximum and average distance from all nodes to any other node
    distances = {}
    for node in network:
        if node.node_id == 1:
            continue
        distances[node.node_id] = {}
        for other_node in network:
            # avoid calculating the distance between a node and itself
            if node.node_id == other_node.node_id:
                continue
            # Avoid calculating the distance between a node and the sink
            if other_node.node_id == 1:
                continue
            distances[node.node_id][other_node.node_id] = network.distance_between_nodes(
                node, other_node)

    # Calculate the average distance per node
    avg_distances = {}
    # Calculate the minimum distance per node
    min_distances = {}
    # Calculate the maximum distance per node
    max_distances = {}
    # Calculate the distance between the node and the sink
    sink_distances = {}
    for node_id, node_distances in distances.items():
        avg_distances[node_id] = sum(node_distances.values()) / \
            len(node_distances)
        min_distances[node_id] = min(node_distances.values())
        max_distances[node_id] = max(node_distances.values())
        sink_distances[node_id] = network.distance_between_nodes(
            network.get_node(node_id), network.get_node(1))

    # Lets put together in an array the average, minimum and maximum distance per node
    avg_min_max_sink_distances = []
    for node_id in avg_distances.keys():
        avg_min_max_sink_distances.extend(
            [avg_distances[node_id], min_distances[node_id], max_distances[node_id], sink_distances[node_id]])

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
                    "energy_dissipated": [round_data['energy_dissipated']],
                    "eelect": [50 * 10**(-9)],
                    "pkt_size": [4000],
                    "eamp": [0.0013 * 10**(-12)],
                    "efs": [10 * 10**(-12)],
                    "eda": [5 * 10**(-9)],
                    "d0": [d0],
                    "avg_min_max_sink_distances": [avg_min_max_sink_distances],
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
        # create the output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df.to_csv(os.path.join(output_folder, "data.csv"), index=False)

    return df


def load_input_data(input_files):
    results = {}
    for input_file in input_files:
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        with open(input_file, 'r') as f:
            data = json.load(f)
            name_parts = file_name.split("_")[1:]
            name_parts[-1] = name_parts[-1].split(".json")[0]
            name = tuple(name_parts)
            name = tuple(float(part.replace("'", ""))
                         for part in name_parts)
            results[name] = data
    print(f'Loaded {len(results)} results')
    return results


def main(args):
    # Load config
    config = load_config(args.config)
    input_files = load_input_data(args.input)
    network = Network(config=config)
    network_model = Extended(config=config, network=network)
    network.set_model(network_model)
    network.initialize()

    process_data(input_files, args.output, network=network)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    parser.add_argument('--input', '-i', type=str,
                        nargs='+', help='Input files', required=True)
    parser.add_argument('--output', '-o', type=str,
                        required=True, help='Output folder')
    args = parser.parse_args()
    # Create the output folder if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)
