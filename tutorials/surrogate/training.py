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

# Lets create a dataloader
BATCH_SIZE = 1024

# Neural network parameters
INPUT_SIZE = 203
HIDDEN_SIZE = 256
OUTPUT_SIZE = 99
LEARNING_RATE = 1e-6
NUM_EPOCHS = 400
PRINT_EVERY = 10
PLOT_EVERY = 1


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
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_size)
        # nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        # x = self.sigmoid(x)
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
        # How many rounds are there?
        max_rounds = len(data)
        # print(f"Name: {name}")
        # Lets loop through the rounds and print the alive nodes
        for round, stats in data.items():
            round = int(round)
            # Lets create a list of the data
            energy_levels = []
            # cluster_heads = []
            membership = []
            for node_id, energy in stats['energy_levels'].items():
                energy_levels.append(energy)
            # Scale the energy levels to 0 to 1, the largest number is 10 J
            # energy_levels = [x / 10 for x in energy_levels]
            # for node_id, cluster_head in stats['cluster_heads'].items():
            #     cluster_heads.append(cluster_head)
            for node_id, cluster_id in stats['membership'].items():
                membership.append(cluster_id)
            # Remove the first element which is the sink
            membership.pop(0)
            # Replace None with 0
            membership = [0 if x is None else x for x in membership]
            # Scale the membership to 0 to 1, the largest number is 100
            # membership = [x / 100 for x in membership]
            # put the name as numbers as they are currently strings
            row = [float(name[0]), float(name[1]), float(name[2]),
                   float(round), float(stats['alive_nodes'])]
            # append the energy levels
            row.extend(energy_levels)
            # append the membership
            row.extend(membership)
            # Y is the membership of the next round
            # Lets check if this is the last round
            if int(round) == max_rounds:
                # If it is then the membership is the same as the current round
                row.extend(membership)
            else:
                # Otherwise the membership is the next round
                # Lets get the next round
                next_round = int(round) + 1
                # Lets get the membership of the next round
                next_round_membership = []
                for node_id, cluster_id in data[str(next_round)]['membership'].items():
                    next_round_membership.append(cluster_id)
                # Remove the first element which is the sink
                next_round_membership.pop(0)
                # Replace None with 0
                next_round_membership = [
                    0 if x is None else x for x in next_round_membership]
                # Scale the membership to 0 to 1, the largest number is 100
                # next_round_membership = [
                #     x / 100 for x in next_round_membership]
                row.extend(next_round_membership)

            data_list.append(row)

    # Split the data into training and testing data
    # Lets use 80% for training and 20% for testing
    # Lets shuffle the data
    random.shuffle(data_list)
    # The X data is the whole data except the last 99 columns which are the membership
    x_data = np.array(data_list)[:, :-99]
    # The y data are the last 99 columns which are the membership
    y_data = np.array(data_list)[:, -99:]

    # Lets save the training and testing data in a json file
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "x_data.json"), "w") as f:
        json.dump(x_data.tolist(), f)
    with open(os.path.join("data", "y_data.json"), "w") as f:
        json.dump(y_data.tolist(), f)

    return x_data, y_data


def main(args):
    if args.data is None:
        X_train, y_train = process_data()
    else:
        # Load the training and testing data
        for file in os.listdir(args.data):
            # Ignore .DS_Store
            if file == ".DS_Store":
                continue
            if file == "x_data.json":
                with open(os.path.join(args.data, file), "r") as f:
                    X_train = np.array(json.load(f))
            elif file == "y_data.json":
                with open(os.path.join(args.data, file), "r") as f:
                    y_train = np.array(json.load(f))
            else:
                print(f"File: {file} is not a valid file.")
                sys.exit(1)

    # Lets one hot encode the y_train
    encoder = OneHotEncoder(sparse=False).fit(y_train)
    # Lets transform the y_train
    y_train = encoder.transform(y_train)

    print(y_train.shape)

    # Accessing the one-hot encoding mapping
    categories = encoder.categories_

    # Printing the one-hot encoding mapping
    for i, category in enumerate(categories):
        print(f'Category {i}: {category}')

    # Split the training data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.7, shuffle=True)

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
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, y_train.shape[1])

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)

    # Lets train the neural network
    # Keep track of losses for plotting
    loss_values = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for seq, y in train_loader:
            optimizer.zero_grad()

            y_pred = model(seq)

            # print(f"y_pred = {y_pred}")
            # input(f"y_train = {y_train}")

            loss = criterion(y_pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')

        # Validation
        if epoch % PRINT_EVERY != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(train_dataset.X)
            train_mse = np.sqrt(criterion(y_pred, train_dataset.y).item())
            y_pred = model(test_dataset.X)
            test_mse = np.sqrt(criterion(y_pred, test_dataset.y).item())
        print(
            f"Epoch: {epoch+1:2} Train RMSE: {train_mse:10.8f} Test RMSE: {test_mse:10.8f}")

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    print("Training Complete")

    step = np.linspace(0, 100, len(loss_values))

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # Lets find out the accuracy of the model
    # Lets loop through the test data
    model.eval()
    accuracy = 0
    total = 0
    with torch.no_grad():
        for seq, y_test in test_loader:
            y_pred = model(seq)
            # Inverse transform the predicted one-hot encoded labels to get categories
            y_pred = encoder.inverse_transform(y_pred)
            print(f"y_pred: {y_pred}")
            # Inverse transform the actual one-hot encoded labels to get categories
            y_test = encoder.inverse_transform(y_test)
            print(f"y_test: {y_test}")
            total += y_test.size(0)
            accuracy += (y_pred == y_test).sum().item()

    print(f"Accuracy: {accuracy / total * 100} %")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", help="Path to the training and testing folder", default=None)
    args = parser.parse_args()
    main(args)
    sys.exit(0)
