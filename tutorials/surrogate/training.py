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
HIDDEN_SIZE = 128
OUTPUT_SIZE = 99
NUM_CLUSTERS = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
PRINT_EVERY = 200
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

    samples = {}

    for name, data in data.items():
        # How many rounds are there?
        max_rounds = len(data)
        samples[name] = {}
        # Lets loop through the rounds and print the alive nodes
        for round, stats in data.items():
            round = int(round)
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
                    membership.append(int(cluster_id))
            # Add a zero for the sink
            membership.insert(0, 0)
            x_data = [float(name[0]), float(name[1]), float(name[2]),
                      float(round), float(stats['alive_nodes'])]
            # append the energy levels
            x_data.extend(energy_levels)
            # Lets check if this is the last round
            if int(round) == max_rounds:
                # If it is then the membership is the same as the current round
                y_data = membership
            else:
                # Otherwise the membership is the next round
                next_round = int(round) + 1
                # Lets get the membership of the next round
                next_round_membership = []
                for node_id, cluster_id in data[str(next_round)]['membership'].items():
                    if cluster_id is None:
                        next_round_membership.append(0)
                    else:
                        # Attach only the cluster id in integer form, remove .0
                        next_round_membership.append(int(cluster_id))
                # Add a zero for the sink
                next_round_membership.insert(0, 0)
                y_data = next_round_membership
            # Lets append the x_data and y_data to the samples
            samples[name][round] = {
                "x_data": x_data,
                "membership": membership,
                "y_data": y_data
            }

    # Lets save the entire dict by name
    os.makedirs("data", exist_ok=True)
    for name, data in samples.items():
        # Lets save the x_data
        with open(os.path.join("data", f"{name}.json"), "w") as f:
            json.dump(data, f)

    return samples


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


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


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
        samples = process_data()
    else:
        # Load the training and testing data
        samples = load_samples()

    print(f"Number of samples: {len(samples)}")

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

    # Print the shape of the training and testing data
    print(f"Shape of the training data: {X_train.shape}, {y_train.shape}")
    print(f"Shape of the testing data: {X_test.shape}, {y_test.shape}")

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

    # weights, current_membership, y_membership = random_sample(samples)

    # Lets create the training data
    # Training_DataSet = NetworkDataset(weights=weights,
    #                                   current_membership=current_membership,
    #                                   y_membership=y_membership)

    # Lets create the dataloader
    train_dataloader = DataLoader(
        Training_DataSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Training_DataSet.collate_fn)

    test_dataloader = DataLoader(
        Testing_DataSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Testing_DataSet.collate_fn)

    # Lets loop through the dataloader
    # for w, x, y in train_dataloader:
    #     print(f"w: {w.shape}, {w.dtype}, x: {x.shape}, {x.dtype}, y: {y.shape}, {y.dtype}")
    #     print(f"w: {w}")
    #     print(f"x: {x}")
    #     print(f"y: {y}")
    #     input("Continue?")

    print(f"Lenght of the dataloader: {len(train_dataloader)}")

    print(
        f"Shape of the training data, x: {Training_DataSet.X.shape}, {Training_DataSet.X.dtype}, y: {Training_DataSet.y.shape}, {Training_DataSet.y.dtype}, weights: {Training_DataSet.weights.shape}, {Training_DataSet.weights.dtype}")

    model = LSTM(embedding_dim=120, vocab_size=101, output_size=101,
                 extra_input_size=Training_DataSet.weights.shape[1])

    # Load the model?
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Variable to save the best model
    best = 0

    # Lets train with the Training_DataSet
    samples_num = 0
    losses = []
    avg_losses = []
    accuracy = []
    model.train()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for weight, input_data, target in train_dataloader:
            samples_num += 1
            X = input_data
            y = target
            w = weight
            model.zero_grad()

            output, (h, c) = model(X, w)

            loss = criterion(output, y)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

         # Calculate the average loss
        avg_loss = sum(losses)/len(losses)
        avg_losses.append(avg_loss)
        losses = []

        # Print the average loss
        print(
            f"EPOCH: {epoch+1}/{NUM_EPOCHS}, avg_loss: {avg_loss:.6f}")

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_weight, test_input, test_output in test_dataloader:
                X = test_input
                y = test_output
                w = test_weight
                output, (h, c) = model(X, w)
                for i in range(output.shape[0]):
                    # Get the predicted
                    _, predicted = torch.max(output[i], 1)
                    # Get the actual
                    actual = y[i]
                    # Get the total
                    total += len(actual)
                    # Get the correct
                    correct += (predicted == actual).sum().item()

        acc = correct/total*100

        print(f"Correct: {correct}, Total: {total} ({acc:.2f}%)")

        accuracy.append(acc)

        if acc > best:
            print(f"New best: {acc:.2f}%")
            best = acc
            # Save the model
            torch.save(model.state_dict(), os.path.join(
                MODELS_PATH, "model.pth"))
        else:
            print(f"Best: {best}%")

    # Lets plot the loss and accuracy
    plt.figure()
    plt.plot(avg_losses)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(MODELS_PATH, "loss.png"))

    plt.figure()
    plt.plot(accuracy)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(MODELS_PATH, "accuracy.png"))


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
