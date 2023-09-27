# This script is used for the training of the surrogate model.
# We use pytorch to train the model.

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


SELF_PATH = os.path.dirname(os.path.abspath(__file__))
# Go to the tutorial folder which is one folder up
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
# Go to the results folder
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")

# Lets create a dataloader
BATCH_SIZE = 16

# Neural network parameters
INPUT_SIZE = 104
HIDDEN_SIZE = 104*2
OUTPUT_SIZE = 100


class NetworkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.layer_2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x


def process_data():
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

    data_list = []

    for name, data in data.items():
        # print(f"Name: {name}")
        # Lets loop through the rounds and print the alive nodes
        for round, stats in data.items():
            # Lets create a list of the data
            energy_levels = []
            # cluster_heads = []
            membership = []
            for node_id, energy in stats['energy_levels'].items():
                energy_levels.append(energy)
            # for node_id, cluster_head in stats['cluster_heads'].items():
            #     cluster_heads.append(cluster_head)
            for node_id, cluster_id in stats['membership'].items():
                membership.append(cluster_id)
            # data_list.append([name[0], name[1], name[2], stats['energy_levels'],
            #                   stats['cluster_heads'], stats['membership'], stats['alive_nodes'], round])
            # put the name as numbers as they are currently strings
            row = [float(name[0]), float(name[1]), float(name[2]),
                   float(round), float(stats['alive_nodes'])]
            # append the energy levels
            row.extend(energy_levels)
            # append the membership
            row.extend(membership)
            data_list.append(row)

    # Split the data into training and testing data
    # Lets use 80% for training and 20% for testing
    # Lets shuffle the data
    random.shuffle(data_list)
    # The X data is the first 3 columns, round, energy levels, alive nodes
    x_data = np.array(data_list)[:, :-100]
    # The y data are the last 100 columns which are the membership
    y_data = np.array(data_list)[:, -100:]
    # Lets split the data
    x_train = x_data[:int(0.8 * x_data.shape[0])]
    y_train = y_data[:int(0.8 * y_data.shape[0])]
    x_test = x_data[int(0.8 * x_data.shape[0]):]
    y_test = y_data[int(0.8 * y_data.shape[0]):]

    # Lets save the training and testing data in a json file
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "x_train.json"), "w") as f:
        json.dump(x_train.tolist(), f)
    with open(os.path.join("data", "y_train.json"), "w") as f:
        json.dump(y_train.tolist(), f)
    with open(os.path.join("data", "x_test.json"), "w") as f:
        json.dump(x_test.tolist(), f)
    with open(os.path.join("data", "y_test.json"), "w") as f:
        json.dump(y_test.tolist(), f)

    return x_train, y_train, x_test, y_test


def main(args):
    if args.data is None:
        X_train, y_train, X_test, y_test = process_data()
    else:
        # Load the training and testing data
        for file in os.listdir(args.data):
            # Ignore .DS_Store
            if file == ".DS_Store":
                continue
            if file == "x_train.json":
                with open(os.path.join(args.data, file), "r") as f:
                    X_train = np.array(json.load(f))
            elif file == "y_train.json":
                with open(os.path.join(args.data, file), "r") as f:
                    y_train = np.array(json.load(f))
            elif file == "x_test.json":
                with open(os.path.join(args.data, file), "r") as f:
                    X_test = np.array(json.load(f))
            elif file == "y_test.json":
                with open(os.path.join(args.data, file), "r") as f:
                    y_test = np.array(json.load(f))
            else:
                print(f"File: {file} is not a valid file.")
                sys.exit(1)

    # Lets load the training data
    train_dataset = NetworkDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)

    # Lets load the testing data
    test_dataset = NetworkDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE, shuffle=True)

    # Check it's working
    for batch, (X, y) in enumerate(train_loader):
        print(f"Batch: {batch+1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        break

    # Lets create the neural network
    model = NeuralNetwork()
    print(model)

    # Lets create the loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Lets train the model
    num_epochs = 2000
    loss_values = []

    for epoch in range(num_epochs):
        for X, y in train_loader:
            print(f"Epoch: {epoch+1}")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

    print("Training Complete")

    step = np.linspace(0, 100, 10500)

    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", help="Path to the training and testing folder", default=None)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
