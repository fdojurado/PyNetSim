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
    def __init__(self, weights, current_membership, y_membership):
        self.weights = torch.from_numpy(weights.astype(np.float32))
        self.X = torch.from_numpy(current_membership.astype(np.int64))
        self.y = torch.from_numpy(y_membership.astype(np.int64))
        self.len = weights.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.weights[idx], self.X[idx], self.y[idx]

    # Support batching
    def collate_fn(self, batch):
        weights = torch.stack([x[0] for x in batch])
        X = torch.stack([x[1] for x in batch])
        y = torch.stack([x[2] for x in batch])
        return weights, X, y


class MixedDataModel(nn.Module):
    def __init__(self, lstm_arch, embedding_dim, num_embeddings, numerical_dim, hidden_dim, lstm_hidden, output_dim, drop_out):
        super(MixedDataModel, self).__init__()

        # Embedding layer for categorical data
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # LSTM layer
        if lstm_arch == "simple":
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden,
                                num_layers=1, batch_first=True)
        if lstm_arch == "complex":
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden,
                                num_layers=2, batch_first=True, dropout=drop_out)
        # Add another layer to enable concatenation with numerical data
        self.lstm_hidden_layer = nn.Linear(
            lstm_hidden, hidden_dim)

        # Combined hidden layer
        self.numerical_layer = nn.Linear(
            numerical_dim, hidden_dim)

        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        # Dropout
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, categorical_data, numerical_data):
        # print(f"Shape of / categorical data 0: {categorical_data.shape}")
        # print(f"First element of categorical data: {categorical_data}")
        # Pass categorical data through embedding layer
        categorical_data = self.embedding_layer(categorical_data)
        # print(f"Shape of / categorical data 1: {categorical_data.shape}")

        # Pass through LSTM layer
        categorical_data, _ = self.lstm(categorical_data)

        categorical_data = self.lstm_hidden_layer(categorical_data)

        categorical_data = self.relu(categorical_data)

        categorical_data = self.dropout(categorical_data)
        # print(f"Shape of / categorical data 2: {categorical_data.shape}")
        # Shape of categorical data: torch.Size([64, 99, 10])
        # print(f"First element of categorical data: {categorical_data[0]}")

        # print(f"Shape of numerical data 0: {numerical_data.shape}")
        # print(f"First element of numerical data: {numerical_data[0]}")

        # pass the numerical data through a linear layer
        numerical_data = self.numerical_layer(numerical_data)

        # print(f"Shape of numerical data 1: {numerical_data.shape}")

        # Pass through the activation function
        numerical_data = self.relu(numerical_data)

        # Dropout
        numerical_data = self.dropout(numerical_data)

        # print(f"Shape of numerical data 2: {numerical_data.shape}")
        # print(f"First element of numerical data: {numerical_data[0]}")

        # Concatenate all the data
        combined_data = torch.cat(
            (categorical_data, numerical_data.unsqueeze(1)), dim=1)
        # input(f"Shape of combined data: {combined_data.shape}")
        # Shape of combined data: torch.Size([1, 99, 201])
        # print(f"First element of combined data: {combined_data[0]}")

        # Pass through hidden layer
        hidden_data = self.hidden_layer(combined_data)

        # Pass through the activation function
        hidden_data = self.relu(hidden_data)

        # dropout
        hidden_data = self.dropout(hidden_data)

        # Pass through output layer
        output_data = self.output_layer(hidden_data)

        # Pass through the activation function
        output_data = self.softmax(output_data)

        return output_data


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
        self.num_workers = self.config.surrogate.num_workers
        self.load_model = self.config.surrogate.load_model
        self.model_path = self.config.surrogate.model_path
        self.raw_data_folder = self.config.surrogate.raw_data_folder
        self.data_folder = self.config.surrogate.data_folder
        self.plots_folder = self.config.surrogate.plots_folder
        self.print_every = self.config.surrogate.print_every
        self.plot_every = self.config.surrogate.plot_every
        self.eval_every = self.config.surrogate.eval_every
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
        logger.info(f"num_workers: {self.num_workers}")
        logger.info(f"load_model: {self.load_model}")
        logger.info(f"model_path: {self.model_path}")
        logger.info(f"raw_data_folder: {self.raw_data_folder}")
        logger.info(f"data_folder: {self.data_folder}")
        logger.info(f"plots_folder: {self.plots_folder}")
        logger.info(f"print_every: {self.print_every}")
        logger.info(f"plot_every: {self.plot_every}")
        logger.info(f"eval_every: {self.eval_every}")
        self.init()

    def init(self):
        # Create the folder to save the model if it does not exist
        # os.makedirs(self.model_path, exist_ok=True)

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
        X_train_weights, X_train_current_membership, Y_train, X_test_weights, X_test_current_membership, Y_test = self.split_data(
            samples)

        # Create the training dataset
        self.training_dataset = NetworkDataset(weights=X_train_weights,
                                               current_membership=X_train_current_membership,
                                               y_membership=Y_train)

        # Create the testing dataset
        self.testing_dataset = NetworkDataset(weights=X_test_weights,
                                              current_membership=X_test_current_membership,
                                              y_membership=Y_test)

        # Create the training dataloader
        self.training_dataloader = DataLoader(
            self.training_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.training_dataset.collate_fn, num_workers=self.num_workers)

        # Create the testing dataloader
        self.testing_dataloader = DataLoader(
            self.testing_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

    def split_data(self, samples):
        # Get all the samples
        x, y, membership = self.get_all_samples(samples)
        np_weights = np.array(x)
        np_weights_size = np_weights.shape
        np_current_membership = np.array(membership)
        np_current_membership_size = np_current_membership.shape
        np_y = np.array(y)

        # Concatenate the weights and current_membership
        np_x = np.concatenate(
            (np_weights, np_current_membership), axis=1)

        if self.test_ratio is None:
            raise Exception("Please provide the test ratio")

        # Lets split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            np_x, np_y, test_size=self.test_ratio, random_state=42, shuffle=False)

        # Lets unpack the weights and current_membership
        X_train_weights = X_train[:, :np_weights_size[1]]
        X_train_current_membership = X_train[:, np_weights_size[1]:]
        Y_train = y_train
        X_test_weights = X_test[:, :np_weights_size[1]]
        X_test_current_membership = X_test[:, np_weights_size[1]:]
        Y_test = y_test

        return X_train_weights, X_train_current_membership, Y_train, X_test_weights, X_test_current_membership, Y_test

    def generate_data(self):
        if self.raw_data_folder is None:
            raise Exception(
                "Please provide the path to the raw data folder to generate the data")
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to save the generated data")
        # Load the data folder
        files = self.load_files(self.raw_data_folder)
        # Normalize the data
        # samples = self.normalize_data(files, normalized_names_values=self.largest_weight,
        #                               normalized_membership_values=1)
        samples = self.normalize_data_cluster_heads(
            files, normalized_names_values=self.largest_weight, normalized_membership_values=1)

        return samples

    def load_data(self):
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to the data folder to load the data")
        # Load the data folder
        samples = self.load_samples(self.data_folder)

        return samples

    def normalize_data_cluster_heads(self, samples, normalized_names_values: int, normalized_membership_values: int):
        normalized_samples = {}
        for name, data in samples.items():
            normalized_samples[name] = {}
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds:
                    continue

                energy_levels = list(stats['energy_levels'].values())
                membership = [0 if cluster_id is None else int(cluster_id) / normalized_membership_values
                              for node_id, cluster_id in stats['membership'].items()]

                # Remove the sink
                # membership = membership[1:]

                # Normalize the cluster ids
                current_membership = membership

                x_data = [value / normalized_names_values for value in name] + \
                    energy_levels

                next_round = round + 1
                next_round_membership = [0 if cluster_id is None else int(
                    cluster_id) / normalized_membership_values for _, cluster_id in data[str(next_round)]['membership'].items()]

                # Get unique cluster ids without including 0
                unique_cluster_ids = set(next_round_membership)
                unique_cluster_ids.remove(0)
                # Convert to list
                unique_cluster_ids = list(unique_cluster_ids)
                # Number of clusters
                num_clusters = len(unique_cluster_ids)
                if num_clusters < 5:
                    # append 0s to make it 5 to the unique cluster ids
                    for _ in range(5 - num_clusters):
                        unique_cluster_ids.append(0)

                # Normalize the cluster ids
                next_round_membership = unique_cluster_ids

                y_data = next_round_membership

                assert all(-1 <= value <=
                           1 for value in x_data), f"Invalid x_data: {x_data}"
                assert all(
                    0 <= value <= self.num_clusters for value in y_data), f"Invalid y_data: {y_data}"
                assert all(
                    0 <= value <= self.num_clusters for value in membership), f"Invalid membership: {membership}"

                normalized_samples[name][str(round)] = {
                    "x_data": x_data,
                    "y_data": y_data,
                    "membership": current_membership
                }

        os.makedirs(self.data_folder, exist_ok=True)
        for name, data in normalized_samples.items():
            with open(os.path.join(self.data_folder, f"{name}.json"), "w") as f:
                json.dump(data, f)

        return normalized_samples

    def normalize_data(self, samples, normalized_names_values: int, normalized_membership_values: int):
        normalized_samples = {}
        for name, data in samples.items():
            normalized_samples[name] = {}
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds:
                    continue

                energy_levels = list(stats['energy_levels'].values())
                membership = [0 if cluster_id is None else int(cluster_id) / normalized_membership_values
                              for node_id, cluster_id in stats['membership'].items()]

                # Add the sink at the beginning
                membership.insert(0, 0)

                x_data = [value / normalized_names_values for value in name] + \
                    energy_levels

                next_round = round + 1
                next_round_membership = [0 if cluster_id is None else int(
                    cluster_id) / normalized_membership_values for node_id, cluster_id in data[str(next_round)]['membership'].items()]

                # Add the sink at the beginning
                next_round_membership.insert(0, 0)

                y_data = next_round_membership

                assert all(-1 <= value <=
                           1 for value in x_data), f"Invalid x_data: {x_data}"
                assert all(
                    0 <= value <= self.num_clusters for value in y_data), f"Invalid y_data: {y_data}"
                assert all(
                    0 <= value <= self.num_clusters for value in membership), f"Invalid membership: {membership}"

                normalized_samples[name][str(round)] = {
                    "x_data": x_data,
                    "y_data": y_data,
                    "membership": membership
                }

        os.makedirs(self.data_folder, exist_ok=True)
        for name, data in normalized_samples.items():
            # with open(os.path.join("data", f"{name}.json"), "w") as f:
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
        # Get the samples in the form of weights, current_membership, y_membership
        x = []
        y = []
        membership = []
        for key, sample in samples.items():
            for round in range(1, len(sample)+1):
                x_data = sample[str(round)]['x_data']
                y_data = sample[str(round)]['y_data']
                pre_membership = sample[str(round)]['membership']
                x.append(x_data)
                y.append(y_data)
                membership.append(pre_membership)
                if len(x_data) != self.numeral_dim:
                    raise (
                        f"Invalid x_data: {key}, {round}, length: {len(x_data)}")

        return x, y, membership

    def get_sample(self, samples, weights: tuple):
        x = []
        y = []
        membership = []
        for key, sample in samples.items():
            if key == weights:
                for round in range(1, len(sample)+1):
                    x_data = sample[str(round)]['x_data']
                    y_data = sample[str(round)]['y_data']
                    pre_membership = sample[str(round)]['membership']
                    x.append(x_data)
                    y.append(y_data)
                    membership.append(pre_membership)
                    if len(x_data) != self.numeral_dim:
                        raise (
                            f"Invalid x_data: {key}, {round}, length: {len(x_data)}")

        return x, y, membership

    def get_model(self, load_model=False):
        model = MixedDataModel(lstm_arch=self.lstm_arch,
                               num_embeddings=self.num_embeddings,
                               embedding_dim=self.embedding_dim,
                               numerical_dim=self.numeral_dim,
                               hidden_dim=self.hidden_dim,
                               lstm_hidden=self.lstm_hidden,
                               output_dim=self.output_dim,
                               drop_out=self.drop_out)

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

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

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
                for input_data, categorical_data, target_data in self.training_dataloader:
                    optimizer.zero_grad()
                    # print(f"Input data: {input_data}, {input_data.shape}")
                    # print(
                    #     f"Categorical data: {categorical_data}, {categorical_data.shape}")
                    # print(f"Target data: {target_data}, {target_data.shape}")
                    outputs = model(categorical_data=categorical_data,
                                    numerical_data=input_data)
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
                    for input_data, categorical_data, target_data in self.testing_dataloader:
                        outputs = model(categorical_data=categorical_data,
                                        numerical_data=input_data)
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

        return model

    def test_predicted_sample(self, y, output, print_output=False):
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == y).sum().item()
        total = np.prod(y.shape)
        if print_output:
            logger.info(f"Y: {y}")
            logger.info(f"Predicted: {predicted}")
            # get the index where the values are equal
            index = np.where(y == predicted)
            logger.info(f"Correct index: {index}")
            # get the index where the values are not equal
            index = np.where(y != predicted)
            logger.info(f"Incorrect index: {index}")
            logger.info(f"Correct: {correct}, Total: {total}")
            input("Press enter to continue")
        return correct, total

    def test(self, batch: int = None, print_output=False, weights: list = None):
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
            x, y, membership = self.get_sample(
                samples, weights=weights)
            # Convert to numpy
            x = np.array(x)
            y = np.array(y)
            membership = np.array(membership)
            self.testing_dataset = NetworkDataset(weights=x,
                                                  current_membership=membership,
                                                  y_membership=y)
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
            for input_data, categorical_data, target_data in self.testing_dataloader:
                X = input_data
                y = target_data
                output = model(categorical_data=categorical_data,
                               numerical_data=X)
                loss = criterion(output, y)
                losses.append(loss.item())
                correct, total = self.test_predicted_sample(
                    y, output, print_output)
                acc = correct/total * 100
                avg_accuracy.append(acc)
        logger.info(f"Average Loss: {np.mean(losses)}")
        logger.info(f"Average Accuracy: {np.mean(avg_accuracy)}")
        logger.info(f"Accuracy Min: {np.min(avg_accuracy)}")
        logger.info(
            f"Number of samples with minimum accuracy: {np.sum(np.array(avg_accuracy) == np.min(avg_accuracy))}")
