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
BATCH_SIZE = 1

# Neural network parameters
INPUT_SIZE = 203
HIDDEN_SIZE = 128
OUTPUT_SIZE = 99
NUM_CLUSTERS = 100
LEARNING_RATE = 1e-6
NUM_EPOCHS = 100
PRINT_EVERY = 100
PLOT_EVERY = 10000


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


class LSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, output_size, extra_input_size=0):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim+extra_input_size,
                            embedding_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(embedding_dim, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, extra_input=None):
        embeds = self.embedding(sequence)
        # Concatenate the extra input
        if extra_input is not None:
            extra_input = extra_input.unsqueeze(1)  # Add batch dimension
            extra_input = extra_input.repeat(
                1, sequence.size(1), 1)  # Repeat for each time step
            embeds = torch.cat((embeds, extra_input), dim=-1)
        # Lets get the lstm
        output, (h, c) = self.lstm(embeds)
        # Lets get the output
        output = self.hidden2tag(output)
        # Lets get the softmax
        output = self.softmax(output)
        return output, (h, c)


def load_samples():
    samples = {}
    for file in os.listdir("data"):
        # Ignore .DS_Store
        if file == ".DS_Store":
            continue
        # Load the data
        with open(os.path.join("data", file), "r") as f:
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
    weights = []
    current_membership = []
    y_membership = []
    for key, sample in samples.items():
        for round in range(1, len(sample)+1):
            x_data = sample[str(round)]['x_data']
            y_data = sample[str(round)]['y_data']
            pre_membership = sample[str(round)]['membership']
            weights.append(x_data)
            current_membership.append(pre_membership)
            y_membership.append(y_data)

    return weights, current_membership, y_membership


def main(args):
    print(f"Loading model: {args.model}")

    # Lets create a dataset

    samples = load_samples()

    # Get all the samples
    weights, current_membership, y_membership = get_all_samples(samples)

    np_weights = np.array(weights)
    np_weights_size = np_weights.shape
    np_current_membership = np.array(current_membership)
    np_current_membership_size = np_current_membership.shape
    np_y = np.array(y_membership)

    # Concatenate the weights and current_membership
    np_x = np.concatenate(
        (np_weights, np_current_membership), axis=1)

    # Lets split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        np_x, np_y, test_size=0.2, random_state=42, shuffle=False)

    # Lets create a dataset
    test_dataset = NetworkDataset(weights=X_test[:, :np_weights_size[1]],
                                  current_membership=X_test[:,
                                                            np_weights_size[1]:],
                                  y_membership=y_test)

    # Create a dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Load the model
    model = LSTM(embedding_dim=120, vocab_size=101, output_size=101,
                 extra_input_size=test_dataset.weights.shape[1])

    model.load_state_dict(torch.load(args.model))

    # Lets test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_weight, test_input, test_output in test_loader:
            X = test_input
            y = test_output
            w = test_weight
            output, (h, c) = model(X, w)
            for i in range(output.shape[0]):
                _, predicted = torch.max(output[i], 1)
                # Get the actual data
                actual = y[i]
                # Get the total
                total = len(actual)
                # Get the correct
                correct = (predicted == actual).sum().item()
                input(f"A: {actual}\nP: {predicted}\nC: {correct}\nT: {total}")

    print(f"Accuracy: {correct/total}")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="The model to use for the testing", required=True
    )
    args = parser.parse_args()
    main(args)
    sys.exit(0)
