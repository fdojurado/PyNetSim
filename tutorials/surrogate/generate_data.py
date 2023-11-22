# Generate the data from JSON files.
import os
import json
import pandas as pd
import argparse
import numpy as np
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


def generate_data_cluster_heads(samples, output_folder, network, export_csv=True):
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
    eelect = 50 * 10**(-9)
    pkt_size = 4000
    eamp = 0.0013 * 10**(-12)
    efs = 10 * 10**(-12)
    eda = 5 * 10**(-9)

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
                eamp_calc = pkt_size*efs*dst**2
            else:
                eamp_calc = pkt_size*eamp*dst**4
            if other_node.node_id == 1:
                tx_energy[node.node_id][other_node.node_id] = (
                    eelect + eda) * pkt_size + eamp_calc
            else:
                tx_energy[node.node_id][other_node.node_id] = eelect * \
                    pkt_size + eamp_calc

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

                # Get the cluster heads of the next round
                cluster_heads = get_round_data(
                    data[str(round+1)])['cluster_heads']
                # Create an array of zeros. As there are 100 nodes, we need 101
                np_cluster_heads = np.zeros(101)
                # Set the index of the cluster heads to 1
                np_cluster_heads[cluster_heads] = 1
                np_cluster_heads = list(np_cluster_heads)
                # Let now get the potential cluster heads
                potential_cluster_heads = []
                # Potential cluster heads are the nodes that have energy above the average of the alive nodes
                energy_levels = round_data['energy_levels']
                # Calculate the average energy of the alive nodes, alive nodes are the nodes that have energy above 0
                nodes_with_energy = [
                    energy_level for energy_level in energy_levels if energy_level > 0]
                if not nodes_with_energy:
                    avg_energy = 0
                else:
                    avg_energy = np.mean(nodes_with_energy)
                # Set the potential cluster heads
                for i, energy_level in enumerate(energy_levels):
                    if energy_level > avg_energy:
                        potential_cluster_heads.append(i+2)
                np_potential_cluster_heads = np.zeros(101)
                np_potential_cluster_heads[potential_cluster_heads] = 1
                np_potential_cluster_heads = list(np_potential_cluster_heads)
                # Lets create a numpy array that contains the estimated energy dissipated when transmitting to the sink
                np_ch_to_sink = np.zeros(101)
                # We only consider the potential cluster heads
                for ch in potential_cluster_heads:
                    # Get from tx_energy the energy dissipated when transmitting from the cluster head to the sink
                    np_ch_to_sink[ch] = tx_energy[ch][1]
                np_ch_to_sink = list(np_ch_to_sink)
                # Lets create a numpy array that contains the estimated energy dissipated when transmitting to the cluster head
                np_non_ch_to_ch = np.zeros(101)
                # Here only consider the nodes that are not cluster heads
                for node in range(2, 101):
                    if node in potential_cluster_heads:
                        continue
                    tx_energy_to_ch = {}
                    for ch in potential_cluster_heads:
                        if ch == node:
                            continue
                        tx_energy_to_ch[ch] = tx_energy[node][ch]
                    tx_energy_to_ch = dict(
                        sorted(tx_energy_to_ch.items(), key=lambda item: item[1]))
                    min_values = list(tx_energy_to_ch.values())[:2]
                    if len(min_values) < 2:
                        if len(min_values) == 1:
                            min_values = [min_values[0]]
                        else:
                            min_values = [0]
                    avg_min_values = np.mean(min_values)
                    np_non_ch_to_ch[node] = avg_min_values
                np_non_ch_to_ch = list(np_non_ch_to_ch)
                # Calculate the estimated energy dissipated by potential cluster heads when receiving data from non cluster heads
                np_ch_from_non_ch = np.zeros(101)
                num_alive_nodes = round_data['alive_nodes']
                estimated_num_chs = int(num_alive_nodes*0.05)
                if estimated_num_chs == 0:
                    estimated_num_chs = 1
                else:
                    num_non_ch_per_ch = num_alive_nodes/estimated_num_chs
                for ch in potential_cluster_heads:
                    ch_rx_energy = eelect*pkt_size*num_non_ch_per_ch
                    np_ch_from_non_ch[ch] = ch_rx_energy
                np_ch_from_non_ch = list(np_ch_from_non_ch)
                # Create a DataFrame for the current round
                df_data = pd.DataFrame({
                    "name": [name],
                    "remaining_energy": [round_data['remaining_energy']],
                    "alive_nodes": [round_data['alive_nodes']],
                    "cluster_heads": [cluster_heads],
                    "cluster_heads_index": [np_cluster_heads],
                    "potential_cluster_heads": [np_potential_cluster_heads],
                    "energy_levels": [round_data['energy_levels']],
                    "energy_dissipated_ch_to_sink": [np_ch_to_sink],
                    "energy_dissipated_non_ch_to_ch": [np_non_ch_to_ch],
                    "energy_dissipated_ch_rx_from_non_ch": [np_ch_from_non_ch],
                    "dst_to_cluster_head": [round_data['dst_to_cluster_head']],
                    "membership": [round_data['membership']],
                    "pdr": [round_data['pdr']],
                    "control_pkt_bits": [round_data['control_pkt_bits']],
                    "pkts_recv_by_bs": [round_data['pkts_recv_by_bs']],
                    "num_cluster_heads": [round_data['num_cluster_heads']],
                    "energy_dissipated": [round_data['energy_dissipated']],
                    "eelect": [eelect],
                    "pkt_size": [pkt_size],
                    "eamp": [eamp],
                    "efs": [efs],
                    "eda": [eda],
                    "d0": [d0],
                    # "avg_min_max_sink_distances": [avg_min_max_sink_distances],
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


def generate_data_cluster_assignment(samples, output_folder, network, export_csv=True):
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
    eelect = 50 * 10**(-9)
    pkt_size = 4000
    eamp = 0.0013 * 10**(-12)
    efs = 10 * 10**(-12)
    eda = 5 * 10**(-9)

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
                eamp_calc = pkt_size*efs*dst**2
            else:
                eamp_calc = pkt_size*eamp*dst**4
            if other_node.node_id == 1:
                tx_energy[node.node_id][other_node.node_id] = (
                    eelect + eda) * pkt_size + eamp_calc
            else:
                tx_energy[node.node_id][other_node.node_id] = eelect * \
                    pkt_size + eamp_calc

    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Processing samples for cluster assignment", total=file_size)

        for name, data in samples.items():
            # print(f"Processing {name}...")
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

                # Get the cluster heads of the next round
                cluster_heads = get_round_data(
                    data[str(round+1)])['cluster_heads']
                # print(f"Cluster heads: {cluster_heads}")
                # Get the membership of the next round
                membership = get_round_data(
                    data[str(round+1)])['membership']
                # print(f"Membership: {membership}")
                # return
                # Lets create a numpy array that contains the estimated energy dissipated when transmitting to the sink
                np_ch_to_sink = np.zeros(101)
                # We only consider the cluster heads
                for ch in cluster_heads:
                    if ch == 0:
                        continue
                    # Get from tx_energy the energy dissipated when transmitting from the cluster head to the sink
                    np_ch_to_sink[ch] = tx_energy[ch][1]
                np_ch_to_sink = list(np_ch_to_sink)
                # print(f"np_ch_to_sink: {np_ch_to_sink}")

                # Lets create a numpy array that contains the estimated energy dissipated when transmitting to the cluster head
                tx_energy_to_ch = {}
                # Here only consider the nodes that are not cluster heads
                for node in range(2, 101):
                    if node in cluster_heads:
                        tx_energy_to_ch[node] = [0, 0, 0, 0, 0]
                        continue
                    # Check if the node has energy
                    if round_data['energy_levels'][node-2] <= 0:
                        tx_energy_to_ch[node] = [0, 0, 0, 0, 0]
                        continue
                    cluster_head_energy = []
                    for ch in cluster_heads:
                        if ch == 0:
                            cluster_head_energy.append(0)
                            continue
                        cluster_head_energy.append(
                            tx_energy[node][ch])

                    tx_energy_to_ch[node] = cluster_head_energy
                    # print len(tx_energy_to_ch[node])
                tx_energy_to_ch_list = [value for _, value in tx_energy_to_ch.items(
                )]
                tx_energy_to_ch_list = [
                    item for sublist in tx_energy_to_ch_list for item in sublist]
                assert len(
                    tx_energy_to_ch_list) == 495, f"len(tx_energy_to_ch_list): {len(tx_energy_to_ch_list)}"
                # print(f"np_non_ch_to_ch: {np_non_ch_to_ch}")
                # print(f"non_ch_closest_ch: {non_ch_closest_ch}")

                # Calculate the estimated energy dissipated by cluster heads when receiving data from non cluster heads
                # np_ch_from_non_ch = np.zeros(101)
                # for ch in cluster_heads:
                #     # count the number of non cluster heads that are assigned to the cluster head
                #     num_non_ch = 0
                #     for _, assigned_ch in non_ch_closest_ch.items():
                #         if assigned_ch == ch:
                #             num_non_ch += 1
                #     # print(f"ch: {ch}, num_non_ch: {num_non_ch}")
                #     ch_rx_energy = eelect*pkt_size*num_non_ch
                #     np_ch_from_non_ch[ch] = ch_rx_energy
                # np_ch_from_non_ch = list(np_ch_from_non_ch)
                # print(f"np_ch_from_non_ch: {np_ch_from_non_ch}")

                # Create a DataFrame for the current round
                df_data = pd.DataFrame({
                    "name": [name],
                    "remaining_energy": [round_data['remaining_energy']],
                    "alive_nodes": [round_data['alive_nodes']],
                    "cluster_heads": [cluster_heads],
                    "energy_levels": [round_data['energy_levels']],
                    "energy_dissipated_ch_to_sink": [np_ch_to_sink],
                    "energy_dissipated_non_ch_to_ch": [tx_energy_to_ch_list],
                    # "energy_dissipated_ch_rx_from_non_ch": [np_ch_from_non_ch],
                    "dst_to_cluster_head": [round_data['dst_to_cluster_head']],
                    "membership": [membership],
                    "pdr": [round_data['pdr']],
                    "control_pkt_bits": [round_data['control_pkt_bits']],
                    "pkts_recv_by_bs": [round_data['pkts_recv_by_bs']],
                    "num_cluster_heads": [round_data['num_cluster_heads']],
                    "energy_dissipated": [round_data['energy_dissipated']],
                    "eelect": [eelect],
                    "pkt_size": [pkt_size],
                    "eamp": [eamp],
                    "efs": [efs],
                    "eda": [eda],
                    "d0": [d0],
                    # "avg_min_max_sink_distances": [avg_min_max_sink_distances],
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

    if args.model == "cluster_heads":
        generate_data_cluster_heads(input_files, args.output, network=network)
    elif args.model == "cluster_assignment":
        generate_data_cluster_assignment(
            input_files, args.output, network=network)
    else:
        raise Exception(f"Unknown model: {args.model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config file", default=CONFIG_FILE)
    parser.add_argument('--input', '-i', type=str,
                        nargs='+', help='Input files', required=True)
    # Generate data for the cluster head or the cluster assignment model?
    parser.add_argument('--model', '-m', type=str,
                        required=True, help="Model to generate data for (cluster_heads or cluster_assignment)")
    parser.add_argument('--output', '-o', type=str,
                        required=True, help='Output folder')
    args = parser.parse_args()
    # Create the output folder if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)
