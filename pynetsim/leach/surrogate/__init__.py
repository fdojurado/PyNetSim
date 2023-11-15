import json
import os
import pandas as pd

from rich.progress import Progress


def print_config(config, **kwargs):
    print("Configuration:")
    print(f"Name: {config.name}")
    print(f"Surrogate: {kwargs.get('surrogate_name', 'unknown')}")
    # print(f"\tLSTM architecture: {config.surrogate.lstm_arch}")
    print(f"\tEpochs: {config.surrogate.epochs}")
    print(f"\tHidden dim: {config.surrogate.hidden_dim}")
    # print(f"\tLSTM hidden: {config.surrogate.lstm_hidden}")
    print(f"\tOutput dim: {config.surrogate.output_dim}")
    print(f"\tNum clusters: {config.surrogate.num_clusters}")
    # print(f"\tNum embeddings: {config.surrogate.num_embeddings}")
    # print(f"\tEmbedding dim: {config.surrogate.embedding_dim}")
    # print(f"\tNumerical dim: {config.surrogate.numerical_dim}")
    print(f"\tWeight decay: {config.surrogate.weight_decay}")
    print(f"\tDrop out: {config.surrogate.drop_out}")
    print(f"\tBatch size: {config.surrogate.batch_size}")
    print(f"\tLearning rate: {config.surrogate.learning_rate}")
    print(f"\tTest ratio: {config.surrogate.test_ratio}")
    print(f"\tLargest weight: {config.surrogate.largest_weight}")
    print(f"\tLargest energy level: {config.surrogate.largest_energy_level}")
    print(f"\tMax dst to ch: {config.surrogate.max_dst_to_ch}")
    print(f"\tNum workers: {config.surrogate.num_workers}")
    print(f"\tLoad model: {config.surrogate.load_model}")
    print(f"\tModel path: {config.surrogate.model_path}")
    print(f"\tGenerate data: {config.surrogate.generate_data}")
    print(f"\tRaw data folder: {config.surrogate.raw_data_folder}")
    print(f"\tData folder: {config.surrogate.data_folder}")
    print(f"\tPlots folder: {config.surrogate.plots_folder}")
    print(f"\tPrint every: {config.surrogate.print_every}")
    print(f"\tPlot every: {config.surrogate.plot_every}")
    print(f"\tEval every: {config.surrogate.eval_every}")
    print(f"Network:")
    print(f"\tNum nodes: {config.network.num_sensor}")


# Method to get a specify fraction of the data for training and testing
def get_data(data, train_fraction=0.8, test_fraction=0.2):
    # Get the number of samples
    num_samples = len(data)

    train_data = data[:int(num_samples * train_fraction)]
    test_data = data[int(num_samples * (1 - test_fraction)):]

    # reset index in both dataframes
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    # print shapes
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, test_data


def load_data(data_url):
    return pd.read_csv(data_url, dtype=str)


def load_files(data_dir):
    samples = {}
    for file in os.listdir(data_dir):
        if file.startswith("LEACH-CE-E_") and not file.endswith("extended.json"):
            name_parts = file.split("_")[1:]
            name_parts[-1] = name_parts[-1].split(".json")[0]
            name = tuple(name_parts)
            name = tuple(float(part.replace("'", ""))
                         for part in name_parts)

            with open(os.path.join(data_dir, file), "r") as f:
                data = json.load(f)
            samples[name] = data
    return samples


def get_round_data(stats):
    energy_levels = list(stats['energy_levels'].values())

    membership = [0 if cluster_id is None else int(cluster_id)
                  for _, cluster_id in stats['membership'].items()]
    # Remove the sink
    membership = membership[1:]

    # Get the remaining energy
    remaining_energy = stats['remaining_energy']

    # Get distance to cluster head
    dst_to_cluster_head = list(stats['dst_to_cluster_head'].values())

    # Get the alive nodes
    alive_nodes = stats['alive_nodes']

    # Get the number of cluster heads
    num_cluster_heads = stats['num_cluster_heads']

    # Get the cluster heads
    if not stats['cluster_heads']:
        cluster_heads = [0] * 5
    else:
        cluster_heads = stats['cluster_heads']
        if len(cluster_heads) < 5:
            cluster_heads += [0] * (5-len(cluster_heads))

    cluster_heads.sort(reverse=False)

    return energy_levels, dst_to_cluster_head, remaining_energy, alive_nodes, cluster_heads, membership


def process_data(samples, data_folder):
    # Lets create a pandas dataframe to store the data
    columns = [
        "alpha", "beta", "gamma", "remaining_energy", "alive_nodes", "cluster_heads", "energy_levels", "dst_to_cluster_head", "membership"]
    df = pd.DataFrame(columns=columns)

    # Get the size of the samples
    file_size = len(samples)

    # Initialize an empty list to store DataFrames
    dfs_list = []

    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Processing samples", total=file_size)

        for name, data in samples.items():
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds - 1:
                    continue

                energy_levels, dst_to_cluster_head, remaining_energy, alive_nodes, cluster_heads, membership = get_round_data(
                    stats)

                # convert the energy levels to a list of integers
                energy_levels = [float(energy_level)
                                 for energy_level in energy_levels]
                dst_to_cluster_head = [float(dst)
                                       for dst in dst_to_cluster_head]
                remaining_energy = float(remaining_energy)
                alive_nodes = int(alive_nodes)
                cluster_heads = [int(cluster_head)
                                 for cluster_head in cluster_heads]
                membership = [int(cluster_id) for cluster_id in membership]
                name = tuple(float(x) for x in name)

                # Create a DataFrame for the current round
                df_data = pd.DataFrame({
                    "alpha": [name[0]],
                    "beta": [name[1]],
                    "gamma": [name[2]],
                    "remaining_energy": [remaining_energy],
                    "alive_nodes": [alive_nodes],
                    "cluster_heads": [cluster_heads],
                    "energy_levels": [energy_levels],
                    "dst_to_cluster_head": [dst_to_cluster_head],
                    "membership": [membership]
                })

                # Check if the dataframe has any nan values
                if df_data.isnull().values.any():
                    raise Exception(f"Dataframe has nan values: {df_data}")

                # Append the DataFrame to the list
                dfs_list.append(df_data)

            progress.update(task, advance=1)

    # Concatenate all DataFrames in the list
    df = pd.concat(dfs_list, ignore_index=True)

    os.makedirs(data_folder, exist_ok=True)
    # Export the df to csv
    df.to_csv(os.path.join(data_folder, "data.csv"), index=False)

    return df


def generate_data(config):
    raw_data_folder = config.surrogate.raw_data_folder
    data_folder = config.surrogate.data_folder
    if raw_data_folder is None:
        raise Exception(
            "Please provide the path to the raw data folder to generate the data")
    if data_folder is None:
        raise Exception(
            "Please provide the path to save the generated data")
    # Load the data folder
    files = load_files(raw_data_folder)
    samples = process_data(files, data_folder)
    return samples
