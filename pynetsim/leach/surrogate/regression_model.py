import os
import json
import math
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


class PolynomialModel(nn.Module):
    def __init__(self, num_weights, hidden_dim):
        super(PolynomialModel, self).__init__()
        # self._degree = degree
        # self.ply = nn.Linear(self._degree, hidden_dim)
        self.weight = nn.Linear(num_weights, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, input_weight):
        # Weights layer
        weight = self.weight(input_weight)

        # output layer
        features = self.output(weight)

        return features

    # def _polynomial_features(self, round):
    #     # print(f"_polynomial_features: {x}, {x.shape}")
    #     return torch.cat([round ** i for i in range(1, self._degree + 1)], 1)


class SurrogateModel:

    def __init__(self, config):
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
        logger.info("Initializing the regression model")
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

        # Create the training dataloader
        self.training_dataloader = DataLoader(
            self.training_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.training_dataset.collate_fn, num_workers=self.num_workers)

        # Create the testing dataset
        self.testing_dataset = NetworkDataset(x=X_test, y=Y_test)

        # Create the testing dataloader
        self.testing_dataloader = DataLoader(
            self.testing_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

    def split_data(self, samples):
        # Get all the samples
        x, y, prev_x, round = self.get_all_samples(samples)
        # print shapes
        np_x = np.array(x)
        np_y = np.array(y)
        np_prev_x = np.array(prev_x)
        np_round = np.array(round)
        # unsqueeze the np_round
        np_round = np.expand_dims(np_round, axis=1)
        print(
            f"np_x: {np_x.shape}, np_y: {np_y.shape}, np_prev_x: {np_prev_x.shape}, np_round: {np_round.shape}")

        # concatenate the x and prev_x
        np_x = np.concatenate((np_x, np_prev_x, np_round), axis=1)

        if self.test_ratio is None:
            raise Exception("Please provide the test ratio")

        # Lets split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            np_x, np_y, test_size=self.test_ratio, random_state=42, shuffle=False)

        # print shapes
        logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def get_all_samples(self, samples):
        x_data_list = []
        prev_x_data_list = []
        y_data_list = []
        round_list = []
        for key, sample in samples.items():
            for round in range(0, len(sample)):
                # if round == 1:
                #     continue
                x_data = sample[str(round)]['x_data']
                prev_x_data = sample[str(round)]['prev_x_data']
                y_data = sample[str(round)]['y_data']
                round = sample[str(round)]['round']
                x_data_list.append(x_data)
                prev_x_data_list.append(prev_x_data)
                y_data_list.append(y_data)
                round_list.append(round)

        return x_data_list, y_data_list, prev_x_data_list, round_list

    def get_sample(self, samples, weights: tuple):
        x_data_list = []
        prev_x_data_list = []
        y_data_list = []
        round_list = []
        for key, sample in samples.items():
            if key == weights:
                for round in range(0, len(sample)):
                    # if round == 1:
                    #     continue
                    x_data = sample[str(round)]['x_data']
                    prev_x_data = sample[str(round)]['prev_x_data']
                    y_data = sample[str(round)]['y_data']
                    round = sample[str(round)]['round']
                    x_data_list.append(x_data)
                    prev_x_data_list.append(prev_x_data)
                    y_data_list.append(y_data)
                    round_list.append(round)

        return x_data_list, y_data_list, prev_x_data_list, round_list

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

    def load_data(self):
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to the data folder to load the data")
        # Load the data folder
        samples = self.load_samples(self.data_folder)

        return samples

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

    def get_round_data(self, name, stats, normalized_names_values: int, normalized_energy_values: int):
        # Get the remaining energy
        remaining_energy = stats['remaining_energy']/200
        assert 0 <= remaining_energy <= 1, f"Invalid remaining energy: {remaining_energy}"

        # Get the alive nodes
        alive_nodes = stats['alive_nodes']/100
        assert 0 <= alive_nodes <= 1, f"Invalid alive nodes: {alive_nodes}"

        # Get the number of cluster heads
        num_cluster_heads = stats['num_cluster_heads']/5
        assert 0 <= num_cluster_heads <= 1, f"Invalid num cluster heads: {num_cluster_heads}"

        # Get the energy levels
        energy_levels = list(stats['energy_levels'].values())
        energy_levels = [
            value / normalized_energy_values for value in energy_levels]
        assert all(-1 <= value <=
                   1 for value in energy_levels), f"Invalid energy levels: {energy_levels}"

        x_data = [value / normalized_names_values for value in name] + \
            [remaining_energy, num_cluster_heads]

        return x_data, alive_nodes, energy_levels

    def generate_data(self):
        if self.raw_data_folder is None:
            raise Exception(
                "Please provide the path to the raw data folder to generate the data")
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to save the generated data")
        # Load the data folder
        files = self.load_files(self.raw_data_folder)

        normalized_samples = {}
        for name, data in files.items():
            normalized_samples[name] = {}
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds-1:
                    continue

                rnd_data, y_data, energy_levels = self.get_round_data(
                    name, stats, self.largest_weight, self.largest_energy_level)

                prev_x_data = []
                prev_round = round-1
                # for prev_round in range(round-1, round-11, -1):
                if prev_round < 0:
                    prev_round_data = [1] * 2
                    # Add the previous energy levels
                    prev_round_data += [1] * 99
                    # Add the previous alive nodes
                    prev_round_data += [y_data]
                    # Add the previous round
                    prev_round_data += [-1]
                else:
                    prev_round_data = normalized_samples[name][str(
                        prev_round)]['x_data']
                    # Remove the first 3 elements
                    prev_round_data = prev_round_data[3:]
                    # Add the previous energy levels
                    prev_round_data += normalized_samples[name][str(
                        prev_round)]['energy_levels']
                    # Add the previous alive nodes
                    prev_round_data += [normalized_samples[name][str(
                        prev_round)]['y_data']]
                    # Add the previous round
                    prev_round_data += [prev_round]
                prev_x_data += prev_round_data

                normalized_samples[name][str(round)] = {
                    'x_data': rnd_data,
                    'prev_x_data': prev_x_data,
                    'y_data': y_data,
                    'round': round/10000,
                    'energy_levels': energy_levels
                }

        os.makedirs(self.data_folder, exist_ok=True)
        for name, data in normalized_samples.items():
            with open(os.path.join(self.data_folder, f"{name}.json"), "w") as f:
                json.dump(data, f)

        return normalized_samples

    def get_model(self, load_model=False):
        model = PolynomialModel(num_weights=109,
                                hidden_dim=self.hidden_dim)

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

        criterion = nn.MSELoss()
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
                    # print(f"Input shape: {input_data.shape}")
                    # print(f"Target shape: {target_data.shape}")
                    target_data = target_data.unsqueeze(1)
                    # print(f"Weight data: {weight_data}, {weight_data.shape}")
                    # print(f"Round data: {round_data}, {round_data.shape}")
                    # print(f"Target data: {target_data}, {target_data.shape}")
                    # print(
                    #     f"Categorical data: {categorical_data}, {categorical_data.shape}")
                    # print(f"Target data: {target_data}, {target_data.shape}")
                    outputs = model(input_weight=input_data)
                    # input(f"Outputs: {outputs}, {outputs.shape}")
                    loss = criterion(outputs, target_data)
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
                        target_data = target_data.unsqueeze(1)
                        outputs = model(
                            input_weight=input_data)
                        loss = criterion(outputs, target_data)
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
                plt.close()

        return model

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
            x, y, prev_x, round = self.get_sample(
                samples, weights=weights)
            np_x = np.array(x)
            np_y = np.array(y)
            np_prev_x = np.array(prev_x)
            np_round = np.array(round)
            # unsqueeze the np_round
            np_round = np.expand_dims(np_round, axis=1)

            # Concatenate the weights and current_membership
            np_x = np.concatenate((np_x, np_prev_x, np_round), axis=1)
            self.testing_dataset = NetworkDataset(x=np_x, y=np_y)
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
        target = []
        predictions = []
        with torch.no_grad():
            for input_data, target_data in self.testing_dataloader:
                target_data = target_data.unsqueeze(1)
                outputs = model(
                    input_weight=input_data)
                loss = criterion(outputs, target_data)
                losses.append(loss.item())
                target.append(target_data.squeeze())
                predictions.append(outputs.squeeze())
        logger.info(f"Average Loss: {np.mean(losses)}")
        if print_output:
            plt.figure()
            plt.plot(target, label="Target")
            plt.plot(predictions, label="Predictions")
            plt.legend()
            plt.savefig(os.path.join(
                self.plots_folder, f"target_predictions.png"))
            plt.close()
