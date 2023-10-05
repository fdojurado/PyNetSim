# This script is used for the training of the surrogate model.
# We use pytorch to train the model.
import math
import time
import sys
import os
import json
import random
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


SELF_PATH = os.path.dirname(os.path.abspath(__file__))
# Go to the tutorial folder which is one folder up
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
# Go to the results folder
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
# Folder to save the models
MODELS_PATH = os.path.join(SELF_PATH, "models")

# Lets create a dataloader
BATCH_SIZE = 64

# Neural network parameters
INPUT_SIZE = 203
HIDDEN_SIZE = 512
OUTPUT_SIZE = 99
NUM_CLUSTERS = 100
LEARNING_RATE = 1e-6
NUM_EPOCHS = 5000
PRINT_EVERY = 100
PLOT_EVERY = 10000


class NetworkDataset(Dataset):
    # Data set that takes input and output data,
    # and supports batching
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def collate_fn(self, batch):
        # Get the data
        x, y = zip(*batch)
        # Stack the data
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y

# Lets create a regression model


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        print(f"Input size: {input_size}, Output size: {output_size}")
        # Define three layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        # Define the activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        # No activation function on the output
        out = self.fc3(out)
        return out


def process_data(normalized_names_values: int = 1, normalized_membership_values: int = 1):
    # Lets loop through all files that start with LEACH-CE-E_ but excluding LEACH-CE-E_extended.json
    data = {}
    for file in os.listdir(RESULTS_PATH):
        if file.startswith("LEACH-CE-E_") and not file.endswith("extended.json"):
            # Get the name only which is after LEACH-CE-E_ and before .json
            # Lets split
            name = file.split("_")[1:]
            name[-1] = name[-1].split(".json")[0]
            # put the name in a tuple
            name = tuple(name)
            # Load the data which is a dictionary
            with open(os.path.join(RESULTS_PATH, file), "r") as f:
                data[name] = json.load(f)

    samples = {}

    for name, data in data.items():
        # How many rounds are there?
        max_rounds = len(data)
        samples[name] = {}
        # Lets loop through the rounds and print the alive nodes
        for round, stats in data.items():
            round = int(round)
            if round == max_rounds:
                continue
            # Lets create a list of the data
            energy_levels = []
            # cluster_heads = []
            membership = []
            for node_id, energy in stats['energy_levels'].items():
                energy_levels.append(energy)
            for node_id, cluster_id in stats['membership'].items():
                if cluster_id is None:
                    membership.append(0)
                else:
                    membership.append(
                        int(cluster_id)/normalized_membership_values)
            # Add a zero for the sink
            membership.insert(0, 0)
            x_data = [float(name[0])/normalized_names_values, float(name[1])/normalized_names_values,
                      float(name[2])/normalized_names_values]
            # append the energy levels
            x_data.extend(energy_levels)
            x_data.extend(membership)
            # Otherwise the membership is the next round
            next_round = round + 1
            # Lets get the membership of the next round
            next_round_membership = []
            for node_id, cluster_id in data[str(next_round)]['membership'].items():
                if cluster_id is None:
                    next_round_membership.append(0)
                else:
                    # Attach only the cluster id in integer form, remove .0
                    next_round_membership.append(
                        int(cluster_id)/normalized_membership_values)
            # Add a zero for the sink
            next_round_membership.insert(0, 0)
            y_data = next_round_membership
            # Lets make sure that x_data and y_data values are between -1 and 1
            # assert max(x_data) <= 1 and min(
            #     x_data) >= -1, f"Max: {max(x_data)}, Min: {min(x_data)}, name: {name}, x_data: {x_data}"
            # assert max(y_data) <= 1 and min(y_data) >= 0
            # Lets append the x_data and y_data to the samples
            samples[name][round] = {
                "x_data": x_data,
                "y_data": y_data
            }

    # Lets save the entire dict by name
    os.makedirs("data_regression", exist_ok=True)
    for name, data in samples.items():
        # Lets save the x_data
        with open(os.path.join("data_regression", f"{name}.json"), "w") as f:
            json.dump(data, f)

    return samples


def load_samples():
    samples = {}
    for file in os.listdir("data_regression"):
        # Ignore .DS_Store
        if file == ".DS_Store":
            continue
        # Load the data
        with open(os.path.join("data_regression", file), "r") as f:
            data = json.load(f)
        # Get the name only which is after LEACH-CE-E_ and before .json
        # Lets get the name which is simply the concatenation of what is inside the ''
        name = file.split(".json")[0]
        name = name.replace("(", "").replace(")", "").split(",")
        name = [x.replace("'", "") for x in name]
        name = [float(x) for x in name]
        name = tuple(name)
        samples[name] = data
    return samples


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_all_samples(samples):
    # Get the samples in the form of weights, current_membership, y_membership
    x = []
    y = []
    for key, sample in samples.items():
        for round in range(1, len(sample)+1):
            x_data = sample[str(round)]['x_data']
            y_data = sample[str(round)]['y_data']
            x.append(x_data)
            y.append(y_data)

    return x, y


def random_sample(samples):
    random_key = random.choice(list(samples.keys()))
    current_membership = []
    y_membership = []
    weights = []
    sample = samples[random_key]
    for round in range(1, len(sample)+1):
        x_data = sample[str(round)]['x_data']
        y_data = sample[str(round)]['y_data']
        pre_membership = sample[str(round)]['membership']
        weights.append(x_data)
        current_membership.append(pre_membership)
        y_membership.append(y_data)

    return weights, current_membership, y_membership


def get_sample_ch(ch, samples):
    # Get a sample where there is a cluster head with id ch
    X_train = []
    y_train = []
    for key, sample in samples.items():
        for round in range(1, len(sample)+1):
            cluster_heads = sample[str(round)]['y_data']
            if ch in cluster_heads:
                y_data = sample[str(round)]['y_data']
                pre_membership = sample[str(round)]['membership']
                x_data = pre_membership
                X_train.append(x_data)
                y_train.append(y_data)
                return X_train, y_train


def main(args):

    # Create a folder to save the model
    os.makedirs("models", exist_ok=True)

    if args.data is None:
        samples = process_data(normalized_names_values=1,
                               normalized_membership_values=1)
    else:
        # Load the training and testing data
        samples = load_samples()

    print(f"Number of samples: {len(samples)}")

    # Get all the samples
    x, y = get_all_samples(samples)

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=False)

    # Lets create a dataset
    train_dataset = NetworkDataset(X_train, y_train)
    test_dataset = NetworkDataset(X_test, y_test)

    # Create a dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test_dataset.collate_fn)

    # Create the model
    model = RegressionModel(
        train_dataset.x.shape[1], HIDDEN_SIZE, test_dataset.y.shape[1])

    # Load the model?
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    # Define the loss function
    criterion = nn.HuberLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lets train the model
    best = float("inf")

    # Lets train the model
    losses = []
    avg_losses = []
    history = []
    avg_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        for input_data, target_data in train_loader:
            X = input_data
            y = target_data
            model.zero_grad()

            # Forward pass
            outputs = model(X)

            # Calculate the loss
            loss = criterion(outputs, y)
            losses.append(loss.item())

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Calculate the average loss
        avg_loss = sum(losses)/len(losses)
        avg_losses.append(avg_loss)
        losses = []

        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

        # Lets validate the model
        model.eval()

        with torch.no_grad():
            for input_data, target_data in test_loader:
                X = input_data
                y = target_data
                outputs = model(X)
                loss = criterion(outputs, y)
                history.append(loss.item())

            # Calculate the average loss
            avg_loss = sum(history)/len(history)
            avg_history.append(avg_loss)
            history = []

            print(
                f"Epoch: {epoch+1}/{NUM_EPOCHS}, Validation Loss: {avg_loss:.4f}")

            # Save the model if the validation loss is the best we've seen so far.
            if avg_loss < best:
                print(f"New best loss: {avg_loss:.4f}")
                best = avg_loss
                torch.save(model.state_dict(), os.path.join(
                    MODELS_PATH, "regression_model.pt"))
            else:
                print(f"Best loss: {best}")

    # Plot the average loss and the average history
    plt.plot(avg_losses, label="Training Loss")
    plt.plot(avg_history, label="Validation Loss")
    plt.legend()
    plt.show()


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
