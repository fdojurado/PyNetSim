import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pynetsim.utils import PyNetSimLogger
from rich.progress import Progress

# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")
PLOTS_PATH = os.path.join(SELF_PATH, "plots")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(namespace=__name__, log_file="my_log.log")
logger = logger_utility.get_logger()


class NetworkDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = x.shape[0]
        logger.info(f"X: {self.X.shape}, y: {self.y.shape}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # Support batching
    def collate_fn(self, batch):
        X = torch.stack([x[0] for x in batch])
        y = torch.stack([x[1] for x in batch])
        return X, y


class ClusterHeadModel(nn.Module):
    def __init__(self):
        super(ClusterHeadModel, self).__init__()

        self.input_layer = nn.Linear(303, 400)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(400, 106)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


class SurrogateModel:

    def __init__(self, config):
        self.name = "Cluster Head Regression Model"
        self.config = config
        self.lstm_arch = config.surrogate.lstm_arch
        self.epochs = self.config.surrogate.epochs
        self.hidden_dim = self.config.surrogate.hidden_dim
        self.lstm_hidden = self.config.surrogate.lstm_hidden
        self.output_dim = self.config.surrogate.output_dim
        self.num_clusters = self.config.surrogate.num_clusters
        self.num_embeddings = self.config.surrogate.num_embeddings
        self.embedding_dim = self.config.surrogate.embedding_dim
        self.numeral_dim = self.config.surrogate.numerical_dim
        self.weight_decay = self.config.surrogate.weight_decay
        self.drop_out = self.config.surrogate.drop_out
        self.batch_size = self.config.surrogate.batch_size
        self.learning_rate = self.config.surrogate.learning_rate
        self.test_ratio = self.config.surrogate.test_ratio
        self.largest_weight = self.config.surrogate.largest_weight
        self.largest_energy_level = self.config.surrogate.largest_energy_level
        self.max_dst_to_ch = self.config.surrogate.max_dst_to_ch
        self.num_workers = self.config.surrogate.num_workers
        self.load_model = self.config.surrogate.load_model
        self.model_path = self.config.surrogate.model_path
        self.raw_data_folder = self.config.surrogate.raw_data_folder
        self.data_folder = self.config.surrogate.data_folder
        self.plots_folder = self.config.surrogate.plots_folder
        self.print_every = self.config.surrogate.print_every
        self.plot_every = self.config.surrogate.plot_every
        self.eval_every = self.config.surrogate.eval_every

    def print_config(self):
        # print all the config values
        logger.info(f"NN model: {self.name}")
        logger.info(f"lstm_arch: {self.lstm_arch}")
        logger.info(f"epochs: {self.epochs}")
        logger.info(f"hidden_dim: {self.hidden_dim}")
        logger.info(f"lstm_hidden: {self.lstm_hidden}")
        logger.info(f"output_dim: {self.output_dim}")
        logger.info(f"num_clusters: {self.num_clusters}")
        logger.info(f"num_embeddings: {self.num_embeddings}")
        logger.info(f"embedding_dim: {self.embedding_dim}")
        logger.info(f"numeral_dim: {self.numeral_dim}")
        logger.info(f"weight_decay: {self.weight_decay}")
        logger.info(f"drop_out: {self.drop_out}")
        logger.info(f"batch_size: {self.batch_size}")
        logger.info(f"learning_rate: {self.learning_rate}")
        logger.info(f"test_ratio: {self.test_ratio}")
        logger.info(f"largest_weight: {self.largest_weight}")
        logger.info(f"largest_energy_level: {self.largest_energy_level}")
        logger.info(f"num_workers: {self.num_workers}")
        logger.info(f"load_model: {self.load_model}")
        logger.info(f"model_path: {self.model_path}")
        logger.info(f"raw_data_folder: {self.raw_data_folder}")
        logger.info(f"data_folder: {self.data_folder}")
        logger.info(f"plots_folder: {self.plots_folder}")
        logger.info(f"print_every: {self.print_every}")
        logger.info(f"plot_every: {self.plot_every}")
        logger.info(f"eval_every: {self.eval_every}")

    def init(self):
        self.print_config()

        # Create the folder to save the plots
        os.makedirs(self.plots_folder, exist_ok=True)

        # if data_path is not provided, then we need to generate the data
        if self.config.surrogate.generate_data:
            samples = self.generate_data()
        else:
            # Load the data
            samples = self.load_data()

        logger.info(f"Number of samples: {len(samples)}")

        # Split the data into training and testing
        X_train, X_test, Y_train, Y_test = self.split_data(samples)

        # Create the training dataset
        self.training_dataset = NetworkDataset(x=X_train, y=Y_train)

        # Create the testing dataset
        self.testing_dataset = NetworkDataset(x=X_test, y=Y_test)

        # Create the training dataloader
        self.training_dataloader = DataLoader(
            self.training_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.training_dataset.collate_fn, num_workers=self.num_workers)

        # Create the testing dataloader
        self.testing_dataloader = DataLoader(
            self.testing_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

    def split_data(self, samples):
        # Get all the samples
        x, y = self.get_all_samples(samples)
        # print shapes
        np_x = np.array(x)
        np_y = np.array(y)

        # print y shape
        logger.info(f"np_y_chs: {np_y.shape}")

        # Create target array with higher likelihoods for nodes in np_y_chs
        np_y_ext = np.zeros(
            (np_y.shape[0], self.num_clusters+1+5))
        for i in range(np_y.shape[0]):
            np_y_ext[i, np_y[i]] = 1

        if self.test_ratio is None:
            raise Exception("Please provide the test ratio")

        # Lets split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            np_x, np_y_ext, test_size=self.test_ratio, random_state=42, shuffle=True)

        # print shapes
        logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def generate_data(self):
        if self.raw_data_folder is None:
            raise Exception(
                "Please provide the path to the raw data folder to generate the data")
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to save the generated data")
        # Load the data folder
        files = self.load_files(self.raw_data_folder)
        samples = self.normalize_data_cluster_heads(
            files, normalized_names_values=self.largest_weight, normalized_membership_values=100,
            normalized_energy_values=self.largest_energy_level, normalized_dst_to_ch_values=self.max_dst_to_ch)

        return samples

    def load_data(self):
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to the data folder to load the data")
        # Load the data folder
        samples = self.load_samples(self.data_folder)

        return samples

    def get_round_data(self, name, stats, normalized_names_values: int, normalized_membership_values: int,
                       normalized_energy_values: int, normalized_dst_to_ch_values: int):
        energy_levels = list(stats['energy_levels'].values())
        energy_levels = [
            value / normalized_energy_values for value in energy_levels]
        assert all(-1 <= value <=
                   1 for value in energy_levels), f"Invalid energy levels: {energy_levels}"
        membership = [0 if cluster_id is None else int(cluster_id) / normalized_membership_values
                      for _, cluster_id in stats['membership'].items()]
        # Remove the sink
        membership = membership[1:]
        assert all(
            0 <= value <= 1 for value in membership), f"Invalid membership: {membership}"

        # Get the remaining energy
        remaining_energy = stats['remaining_energy']/400
        assert 0 <= remaining_energy <= 1, f"Invalid remaining energy: {remaining_energy}"

        # Get distance to cluster head
        dst_to_cluster_head = [
            value / normalized_dst_to_ch_values for value in stats['dst_to_cluster_head'].values()]
        assert all(0 <= value <=
                   1 for value in dst_to_cluster_head), f"Invalid dst to cluster head: {dst_to_cluster_head}"

        # Get the alive nodes
        alive_nodes = stats['alive_nodes']/100
        assert 0 <= alive_nodes <= 1, f"Invalid alive nodes: {alive_nodes}"

        # Get the number of cluster heads
        num_cluster_heads = stats['num_cluster_heads']/5
        assert 0 <= num_cluster_heads <= 1, f"Invalid num cluster heads: {num_cluster_heads}"

        x_data = [value / normalized_names_values for value in name] + \
            energy_levels + \
            dst_to_cluster_head + \
            [remaining_energy, alive_nodes, num_cluster_heads] +\
            membership

        return x_data

    def normalize_data_cluster_heads(self, samples, normalized_names_values: int, normalized_membership_values: int,
                                     normalized_energy_values: int, normalized_dst_to_ch_values: int):
        normalized_samples = {}
        for name, data in samples.items():
            normalized_samples[name] = {}
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds-1:
                    continue

                rnd_data = self.get_round_data(
                    name, stats, normalized_names_values, normalized_membership_values, normalized_energy_values, normalized_dst_to_ch_values)

                next_round = round + 1

                # Get the cluster heads
                if not data[str(next_round)]['cluster_heads']:
                    next_round_chs = []
                    for i in range(5):
                        next_round_chs.append(self.num_clusters+i+1)
                else:
                    next_round_chs = data[str(next_round)]['cluster_heads']
                    count = 1
                    while len(next_round_chs) < 5:
                        value = self.num_clusters+count
                        if value not in next_round_chs:
                            next_round_chs.append(value)
                        count += 1

                assert len(
                    next_round_chs) == 5, f"Invalid chs: {next_round_chs}"

                normalized_samples[name][str(round)] = {
                    "x_data": rnd_data,
                    "y_chs": next_round_chs,
                }

        os.makedirs(self.data_folder, exist_ok=True)
        for name, data in normalized_samples.items():
            with open(os.path.join(self.data_folder, f"{name}.json"), "w") as f:
                json.dump(data, f)

        return normalized_samples

    def load_files(self, data_dir):
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

    def load_samples(self, data_dir):
        logger.info(f"Loading samples from: {data_dir}")
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

    def get_all_samples(self, samples):
        x_data_list = []
        y_chs_list = []
        for _, sample in samples.items():
            for round in range(0, len(sample)):
                x_data = sample[str(round)]['x_data']
                y_chs = sample[str(round)]['y_chs']
                x_data_list.append(x_data)
                y_chs_list.append(y_chs)

        return x_data_list, y_chs_list

    def get_sample(self, samples, weights: tuple):
        x_data_list = []
        y_chs_list = []
        for key, sample in samples.items():
            if key == weights:
                for round in range(0, len(sample)):
                    x_data = sample[str(round)]['x_data']
                    y_chs = sample[str(round)]['y_chs']
                    x_data_list.append(x_data)
                    y_chs_list.append(y_chs)

        return x_data_list, y_chs_list

    def get_model(self, load_model=False):
        model = ClusterHeadModel()

        if self.load_model:
            if self.model_path is None:
                raise Exception(
                    "Please provide the path to the model to load")
            logger.info(f"Loading model: {self.model_path}")
            model.load_state_dict(torch.load(self.model_path))
        else:
            if load_model:
                raise Exception(
                    "Please provide the path to the model to load")
            logger.info(f"Creating new model")
            # Lets make sure that the folder to save the model exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate)

        return model, criterion, optimizer

    def train(self):
        if self.model_path is None:
            raise Exception("Please provide the path to save the model")

        model, criterion, optimizer = self.get_model()

        best_loss = float("inf")
        train_losses = []
        validation_losses = []

        for epoch in range(self.epochs):
            model.train()
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Training (epoch {epoch}/{self.epochs})", total=len(self.training_dataloader))
                for input_data, target_data in self.training_dataloader:
                    optimizer.zero_grad()
                    chs = model(
                        x=input_data)
                    loss = criterion(chs, target_data)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                    progress.update(task, advance=1)

            avg_train_loss = np.mean(train_losses)

            if epoch % self.print_every == 0:
                logger.info(
                    f"Epoch [{epoch}/{self.epochs}] Train Loss: {avg_train_loss:.4f}")

            if epoch % self.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for input_data, target_data in self.testing_dataloader:
                        chs = model(
                            x=input_data)
                        loss = criterion(chs, target_data)
                        validation_losses.append(loss.item())
                avg_val_loss = np.mean(validation_losses)
                if avg_val_loss < best_loss:
                    logger.info(
                        f"Epoch [{epoch}/{self.epochs}] Validation Loss Improved: {best_loss:.4f} -> {avg_val_loss:.4f}"
                    )
                    best_loss = avg_val_loss
                    if self.model_path:
                        torch.save(model.state_dict(), self.model_path)

            if epoch % self.plot_every == 0:
                plt.figure()  # Create a new figure
                plt.plot(train_losses, label="Train Loss")
                plt.plot(validation_losses, label="Validation Loss")
                plt.legend()
                plt.savefig(os.path.join(
                    self.plots_folder, f"train_validation_loss_classification.png"))

        return model

    def test_predicted_sample(self, y, output, print_output=False):
        _, y_indices = torch.topk(y, k=5, dim=1)
        _, topk_indices = torch.topk(output, k=5, dim=1)
        predicted = topk_indices
        # to numpy
        predicted = predicted.numpy()
        # Sort the predicted
        predicted.sort()
        y = y_indices.numpy()
        # sort the y
        y.sort()
        # How many y values are in the predicted
        correct = 0
        total = 0
        for i in range(len(y)):
            # how many values are in the predicted
            correct += np.sum(np.isin(y[i], predicted[i]))
            total += len(y[i])
            if print_output:
                logger.info(f"Y: {y[i]}, Predicted: {predicted[i]}")
                # get the index where the values are equal
                index = np.where(y[i] == predicted[i])
                logger.info(f"Correct index: {index}")
                # get the index where the values are not equal
                index = np.where(y[i] != predicted[i])
                logger.info(f"Incorrect index: {index}")
                logger.info(f"Correct: {correct}, Total: {total}")
                input("Press enter to continue")
        return correct, total

    def test(self, batch: int = None, print_output=False, weights: list = None):
        logger.info(f"Testing with batch size: {batch}, weights: {weights}")
        # Lets check if the path to the model exists
        if self.model_path is None:
            raise Exception("Please provide the path to the model")

        model, criterion, _ = self.get_model(load_model=True)

        if weights is not None:
            if batch is not None:
                self.batch_size = batch
            # Convert weights to tuple
            weights = tuple(weights)
            # Load the data
            samples = self.load_data()
            x, y = self.get_sample(
                samples, weights=weights)
            np_x = np.array(x)
            np_y = np.array(y)

            # Create target array with higher likelihoods for nodes in np_y_chs
            np_y_ext = np.zeros(
                (np_y.shape[0], self.num_clusters+1+5))
            for i in range(np_y.shape[0]):
                np_y_ext[i, np_y[i]] = 1

            self.testing_dataset = NetworkDataset(x=np_x, y=np_y_ext)
            # recreate the dataloader
            self.testing_dataloader = DataLoader(
                self.testing_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

        elif batch is not None:
            self.batch_size = batch
            # recreate the dataloader
            self.testing_dataloader = DataLoader(
                self.testing_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

        model.eval()
        losses = []
        avg_accuracy = []
        with torch.no_grad():
            for input_data, target_data in self.testing_dataloader:
                # Print the X every 209 elements in the array
                # temp = X[0]
                # for i in range(0, len(temp), 209):
                #     print(f"X {i/209}: {temp[i:i+209]}")
                # print(f"Input data: {input_data}, shape: {input_data.shape}")
                # print(
                #     f"Target data: {target_data}, shape: {target_data.shape}")
                # _, indices = torch.topk(target_data, k=5, dim=1)
                # indices = indices.numpy()
                # indices.sort()
                # print(f"Indices: {indices}")
                chs = model(x=input_data)
                loss = criterion(chs, target_data)
                losses.append(loss.item())
                correct, total = self.test_predicted_sample(
                    target_data, chs, print_output)
                acc = correct/total * 100
                avg_accuracy.append(acc)
        logger.info(f"Average Loss: {np.mean(losses)}")
        logger.info(f"Average Accuracy: {np.mean(avg_accuracy)}")
        logger.info(f"Accuracy Min: {np.min(avg_accuracy)}")
        logger.info(
            f"Number of samples with minimum accuracy: {np.sum(np.array(avg_accuracy) == np.min(avg_accuracy))}")
