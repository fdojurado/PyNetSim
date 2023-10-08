import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rich.logging import RichHandler
from pynetsim.utils import PyNetSimLogger

# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")
PLOTS_PATH = os.path.join(SELF_PATH, "plots")

# Configuration parameters
BATCH_SIZE = 32
INPUT_SIZE = 203
HIDDEN_SIZE = 256
OUTPUT_SIZE = 99
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5000
LARGEST_WEIGHT = 6
NUM_CLUSTERS = 100

# Print and plot intervals
PRINT_EVERY = 1
PLOT_EVERY = 10
PLOT_EVERY = 10000

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log")
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
    def __init__(self, embedding_dim, num_embeddings, numerical_dim, hidden_dim, output_dim):
        super(MixedDataModel, self).__init__()

        # Embedding layer for categorical data
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # print(
        #     f"Size of the embedding layer, input: {num_embeddings}, output: {embedding_dim}")

        # Combined hidden layer
        self.hidden_layer = nn.Linear(
            numerical_dim+embedding_dim, hidden_dim)

        # print(
        #     f"Size of the hidden layer, input: {numerical_dim+embedding_dim}, output: {hidden_dim}")

        # Add another hidden layer
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # print(f"Size of the hidden layer2, input: {hidden_dim}, output: {hidden_dim}")

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # print(f"Size of the output layer, input: {256}, output: {output_dim}")

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=2)

        # Dropout
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, categorical_data, numerical_data):
        # print(f"Shape of / categorical data 0: {categorical_data.shape}")
        # print(f"First element of categorical data: {categorical_data[0]}")
        # Pass categorical data through embedding layer
        categorical_data = self.embedding_layer(categorical_data)
        # print(f"Shape of / categorical data: {categorical_data.shape}")
        # Shape of categorical data: torch.Size([64, 99, 10])
        # print(f"First element of categorical data: {categorical_data[0]}")

        # print(f"Shape of numerical data 0: {numerical_data.shape}")
        # print(f"First element of numerical data: {numerical_data[0]}")

        # Expand dimensions of numerical data to match the sequence length
        numerical_data = numerical_data.unsqueeze(
            1).expand(-1, categorical_data.size(1), -1)
        # print(f"Shape of numerical data 0: {numerical_data.shape}")
        # print(f"First element of numerical data: {numerical_data[0]}")

        # Concatenate all the data
        combined_data = torch.cat(
            (categorical_data, numerical_data), dim=2)
        # input(f"Shape of combined data: {combined_data.shape}")
        # Shape of combined data: torch.Size([1, 99, 201])
        # input(f"First element of combined data: {combined_data[0]}")

        # Pass through hidden layer
        hidden_data = self.hidden_layer(combined_data)

        # Pass through the activation function
        hidden_data = self.relu(hidden_data)

        # dropout
        # hidden_data = self.dropout(hidden_data)

        # Pass through hidden layer
        hidden_data = self.hidden_layer2(hidden_data)

        # Pass through the activation function
        hidden_data = self.relu(hidden_data)

        # dropout
        # hidden_data = self.dropout(hidden_data)

        # Pass through output layer
        output_data = self.output_layer(hidden_data)

        # Pass through the activation function
        output_data = self.softmax(output_data)

        # input(f"Shape of output data: {output_data.shape}")

        return output_data


def normalize_data(samples, normalized_names_values=LARGEST_WEIGHT, normalized_membership_values=NUM_CLUSTERS):
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
                0 <= value <= NUM_CLUSTERS for value in y_data), f"Invalid y_data: {y_data}"
            assert all(
                0 <= value <= NUM_CLUSTERS for value in membership), f"Invalid membership: {membership}"

            normalized_samples[name][round] = {
                "x_data": x_data,
                "y_data": y_data,
                "membership": membership
            }

    os.makedirs(os.path.join(SELF_PATH, "data"), exist_ok=True)
    for name, data in normalized_samples.items():
        # with open(os.path.join("data", f"{name}.json"), "w") as f:
        with open(os.path.join(SELF_PATH, f"data/{name}.json"), "w") as f:
            json.dump(data, f)

    return normalized_samples


def load_files(data_dir):
    samples = {}
    for file in os.listdir(data_dir):
        if file.startswith("LEACH-CE-E_") and not file.endswith("extended.json"):
            name_parts = file.split("_")[1:]
            name_parts[-1] = name_parts[-1].split(".json")[0]
            name = tuple(name_parts)
            name = tuple(float(part.replace("'", "")) for part in name_parts)

            with open(os.path.join(data_dir, file), "r") as f:
                data = json.load(f)
            samples[name] = data
    return samples


def load_samples(data_dir):
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


def get_model(learning_rate=LEARNING_RATE, load_model=None):
    model = MixedDataModel(num_embeddings=101,
                           embedding_dim=50,
                           numerical_dim=102,
                           hidden_dim=512,
                           output_dim=101)

    if load_model:
        logger.info(f"Loading model: {load_model}")
        model.load_state_dict(torch.load(load_model))

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_model(load_model, train_loader, test_loader, input_size, hidden_size, output_size, num_epochs, learning_rate, model_path=None):
    # print(
    #     f"Input size: {input_size}, hidden size: {hidden_size}, output size: {output_size}")

    model, criterion, optimizer = get_model(learning_rate=learning_rate,
                                            load_model=load_model)

    best_loss = float("inf")
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        for input_data, categorical_data, target_data in train_loader:
            optimizer.zero_grad()
            # print(f"Input data: {input_data}, {input_data.shape}")
            # print(f"Target data: {target_data}, {target_data.shape}")
            outputs = model(categorical_data=categorical_data,
                            numerical_data=input_data)
            # input(f"Outputs: {outputs}, {outputs.shape}")
            loss = criterion(outputs, target_data)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for input_data, categorical_data, target_data in test_loader:
                outputs = model(categorical_data=categorical_data,
                                numerical_data=input_data)
                loss = criterion(outputs, target_data)
                validation_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(validation_losses)

        if epoch % PRINT_EVERY == 0:
            logger.info(
                f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}")

        if epoch % PLOT_EVERY == 0:
            plt.figure()  # Create a new figure
            plt.plot(train_losses, label="Train Loss")
            plt.plot(validation_losses, label="Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(
                PLOTS_PATH, f"train_validation_loss_classification.png"))

        if avg_val_loss < best_loss:
            logger.info(
                f"Epoch [{epoch}/{num_epochs}] Validation Loss Improved: {best_loss:.4f} -> {avg_val_loss:.4f}"
            )
            best_loss = avg_val_loss
            if model_path:
                torch.save(model.state_dict(), model_path)

    return model


def get_all_samples(samples):
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
            if len(x_data) != 102:
                raise (
                    f"Invalid x_data: {key}, {round}, length: {len(x_data)}")

    return x, y, membership


def main(args):

    # Create a folder to save the model
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Plot folder
    os.makedirs(PLOTS_PATH, exist_ok=True)

    if args.data is None:
        files = load_files(RESULTS_PATH)
        samples = normalize_data(
            files, normalized_names_values=LARGEST_WEIGHT, normalized_membership_values=1)
    else:
        # Load the training and testing data
        samples = load_samples(args.data)

    logger.info(f"Number of samples: {len(samples)}")

    # Get all the samples
    x, y, membership = get_all_samples(samples)

    # Print the shape of the data
    # print(f"Length of x: {len(x)}")
    # print(f"Length of y: {len(y)}")
    # print(f"Length of membership: {len(membership)}")

    np_weights = np.array(x)
    np_weights_size = np_weights.shape
    np_current_membership = np.array(membership)
    np_current_membership_size = np_current_membership.shape
    np_y = np.array(y)

    # Concatenate the weights and current_membership
    np_x = np.concatenate(
        (np_weights, np_current_membership), axis=1)

    # Lets split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        np_x, np_y, test_size=0.2, random_state=42, shuffle=False)

    # Print the shape of the training and testing data
    # print(f"Shape of the training data: {X_train.shape}, {y_train.shape}")
    # print(f"Shape of the testing data: {X_test.shape}, {y_test.shape}")

    # Lets unpack the weights and current_membership
    X_train_weights = X_train[:, :np_weights_size[1]]
    X_train_current_membership = X_train[:, np_weights_size[1]:]

    # Lets create the training data
    Training_DataSet = NetworkDataset(weights=X_train_weights,
                                      current_membership=X_train_current_membership,
                                      y_membership=y_train)

    # Lets create the testing data
    Testing_DataSet = NetworkDataset(weights=X_test[:, :np_weights_size[1]],
                                     current_membership=X_test[:,
                                                               np_weights_size[1]:],
                                     y_membership=y_test)

    # Lets create the dataloader
    train_dataloader = DataLoader(
        Training_DataSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Training_DataSet.collate_fn)

    test_dataloader = DataLoader(
        Testing_DataSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Testing_DataSet.collate_fn)

    logger.info(f"Lenght of the dataloader: {len(train_dataloader)}")

    logger.info(
        f"Shape of the training data, x: {Training_DataSet.X.shape}, {Training_DataSet.X.dtype}, y: {Training_DataSet.y.shape}, {Training_DataSet.y.dtype}, weights: {Training_DataSet.weights.shape}, {Training_DataSet.weights.dtype}")

    model_path = os.path.join(MODELS_PATH, "model.pt")

    # train_model(args.load, train_dataloader, test_dataloader, Training_DataSet.X.shape[1],
    #             HIDDEN_SIZE, Training_DataSet.y.shape[1], NUM_EPOCHS, LEARNING_RATE, model_path)
    train_model(load_model=args.load, train_loader=train_dataloader, test_loader=test_dataloader, input_size=Training_DataSet.weights.shape[1],
                hidden_size=HIDDEN_SIZE, output_size=Training_DataSet.y.shape[1], num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, model_path=model_path)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", help="Path to the training and testing folder", default=None)
    # Load model?
    parser.add_argument(
        "-l", "--load", help="Path to the model to load", default=None)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
