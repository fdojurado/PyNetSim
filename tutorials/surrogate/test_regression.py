# This script is used for the testing of the regression model
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
LEARNING_RATE = 1e-46
NUM_EPOCHS = 100
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


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
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


def closest_value(output):
    # Convert tensors to numpy
    output = np.array(output)

    for i in range(len(output)):
        # Get the closest value
        closest = np.round(output[i])
        # Convert to integer
        output[i] = closest.astype(int)
    return output


def test_predicted_sample(x, y, output):
    # print(f"Shape of x: {x.shape}")
    # print(f"Shape of y: {y.shape}")
    # print(f"Shape of output: {output.shape}")

    # For each element in the y, find the closest integer
    y = closest_value(y)
    output = closest_value(output)
    # input(f"closes y: {y}")
    # input(f"closest output: {output}")
    # Convert to numpy
    y = np.array(y)
    output = np.array(output)
    # Count the number of correct predictions
    correct = np.sum(y == output)
    # Count the total number of predictions
    total = np.prod(y.shape)
    # Get the index of the correct predictions
    # correct_index = np.where(y == output)
    # print(f"correct_index {correct_index}")

    # input(
    #     f"Correct: {correct}, Total: {total}, Accuracy: {correct/total*100:.2f}%")

    return correct, total


def test_predicted_batch(x, y, output):
    # print shapes
    # print(f"X: {x.shape}")
    # print(f"Y: {y.shape}")
    # print(f"Output: {output.shape}")
    accuracy = []

    # Loop through the batch
    for i in range(len(output)):
        # Get the ith output
        ith_output = output[i]
        # Get the ith y
        ith_y = y[i]
        # Get the ith x
        ith_x = x[i]
        correct, total = test_predicted_sample(
            ith_x, ith_y, ith_output)

        accuracy.append(correct/total*100)

    # Get the average accuracy
    average_accuracy = np.mean(accuracy)

    return average_accuracy


def main(args):
    print(f"Loading model: {args.model}")

    # Lets create a dataset

    samples = load_samples()

    # Get all the samples
    x, y = get_all_samples(samples)

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=False)

    # Lets create a dataset
    test_dataset = NetworkDataset(X_test, y_test)

    # Create a dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Load the model
    model = RegressionModel(
        test_dataset.x.shape[1], HIDDEN_SIZE, test_dataset.y.shape[1])

    # Define the loss function
    criterion = nn.HuberLoss()

    model.load_state_dict(torch.load(args.model))

    # Lets test the model
    model.eval()
    losses = []
    avg_accuracy = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            X = batch_x
            y = batch_y
            # print shapes
            # print(f"X: {X.shape}")
            # print(f"Y: {y.shape}")
            # Get the output
            output = model(X)

            if BATCH_SIZE == 1:
                # input(f"testing prediction")
                test_predicted_sample(X, y, output)
                continue

            loss = criterion(output, y)
            losses.append(loss.item())
            acc = test_predicted_batch(X, y, output)
            avg_accuracy.append(acc)

    # Print the average loss
    print(f"Average Loss: {np.mean(losses)}")
    # Print the average accuracy
    print(f"Average Accuracy: {np.mean(avg_accuracy)}")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="The model to use for the testing", required=True
    )
    args = parser.parse_args()
    main(args)
    sys.exit(0)
