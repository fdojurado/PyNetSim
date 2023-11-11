import json
import os


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


def load_samples(data_dir):
    print(f"Loading samples from: {data_dir}")
    samples = {}
    for file in os.listdir(data_dir):
        if file == ".DS_Store":
            continue
        with open(os.path.join(data_dir, file), "r") as f:
            data = json.load(f)
        # Remove single quotes and split by comma
        name = tuple(float(x.replace("'", "")) for x in file.split(
            ".json")[0].replace("(", "").replace(")", "").split(","))
        samples[name] = data
    return samples


def load_data(data_folder):
    if data_folder is None:
        raise Exception(
            "Please provide the path to the data folder to load the data")
    # Load the data folder
    samples = load_samples(data_folder)
    return samples


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


def generate_data(func, config):
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
    samples = func(files, config)
    return samples
