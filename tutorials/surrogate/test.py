import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")
HIDDEN_SIZE = 512
NUM_EPOCHS = 100


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

        # Other input layers for numerical and text data
        self.numerical_layer = nn.Linear(numerical_dim, hidden_dim)

        # print(
        #     f"Size of the input of the hidden layer: {num_embeddings + hidden_dim}")
        # print(
        #     f"Size of the input of the hidden layer2: {embedding_dim + hidden_dim}")
        # print(
        #     f"Size of the input of the hidden layer3: {num_embeddings+numerical_dim}")
        # Combined hidden layer
        self.hidden_layer = nn.Linear(
            embedding_dim+numerical_dim, hidden_dim)

        # Add another hidden layer
        self.hidden_layer2 = nn.Linear(hidden_dim, 128)

        # Output layer
        self.output_layer = nn.Linear(128, output_dim)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, categorical_data, numerical_data):
        # print(f"Shape of / categorical data 0: {categorical_data.shape}")
        # Pass categorical data through embedding layer
        categorical_data = self.embedding_layer(categorical_data)
        # print(f"Shape of / categorical data: {categorical_data.shape}")
        # Shape of categorical data: torch.Size([1, 99, 99])

        # Expand dimensions of numerical data to match the sequence length
        numerical_data = numerical_data.unsqueeze(
            1).expand(-1, categorical_data.size(1), -1)
        # print(f"Shape of numerical data: {numerical_data.shape}")
        # Shape of numerical data: torch.Size([1, 99, 102])

        # Concatenate all the data
        combined_data = torch.cat(
            (categorical_data, numerical_data), dim=2)
        # print(f"Shape of combined data: {combined_data.shape}")
        # Shape of combined data: torch.Size([1, 99, 201])

        # Pass through hidden layer
        hidden_data = self.hidden_layer(combined_data)

        # Pass through the activation function
        hidden_data = self.relu(hidden_data)

        # Pass through hidden layer
        hidden_data = self.hidden_layer2(hidden_data)

        # Pass through the activation function
        hidden_data = self.relu(hidden_data)

        # Pass through output layer
        output_data = self.output_layer(hidden_data)

        # Pass through the activation function
        output_data = self.softmax(output_data)

        return output_data


def load_samples():
    samples = {}
    data_dir = os.path.join(SELF_PATH, "data")
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


def test_predicted_sample(y, output, print_output=False):
    # input(f"Y: {y}, shape: {y.shape}")
    # input(f"Shape of output: {output.shape}")
    _, predicted = torch.max(output.data, 1)
    # input(f"Predicted: {predicted}")
    correct = (predicted == y).sum().item()
    total = np.prod(y.shape)
    # input(f"Correct: {correct}, Total: {total}")
    if print_output:
        print(f"Y: {y}")
        print(f"Predicted: {predicted}")
        # get the index where the values are equal
        index = np.where(y == predicted)
        print(f"Index: {index}")
        print(f"Correct: {correct}, Total: {total}")
    return correct, total


def test_predicted_batch(y, output_batch, print_output=False):
    accuracy = []
    # input(f"Shape of output: {output_batch.shape}")
    # Loop through the batch [64, 100, 100]
    # We want to loop through the [1, 100, 100] and compare it with the y
    for i in range(len(output_batch)):
        # input(f"Output: {output_batch[i]}, shape: {output_batch[i].shape}")
        correct, total = test_predicted_sample(
            y[i], output_batch[i], print_output)
        accuracy.append(correct/total * 100)
    return np.mean(accuracy)


def main(args):
    print(f"Loading model: {args.model}")
    samples = load_samples()
    x, y, membership = get_all_samples(samples)

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
    # Lets unpack the weights and current_membership
    X_test_weights = X_test[:, :np_weights_size[1]]
    X_test_current_membership = X_test[:, np_weights_size[1]:]

    test_dataset = NetworkDataset(weights=X_test_weights,
                                  current_membership=X_test_current_membership,
                                  y_membership=y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collate_fn)
    model = MixedDataModel(num_embeddings=200,
                           embedding_dim=10,
                           numerical_dim=102,
                           hidden_dim=512,
                           output_dim=101)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    losses = []
    avg_accuracy = []
    with torch.no_grad():
        for input_data, categorical_data, target_data in test_loader:
            X = input_data
            y = target_data
            output = model(categorical_data=categorical_data,
                           numerical_data=X)
            loss = criterion(output, y)
            losses.append(loss.item())
            if args.batch == 1:
                correct, total = test_predicted_sample(y, output, args.print)
                avg_accuracy.append(correct/total*100)
                continue
            acc = test_predicted_batch(y, output, args.print)
            avg_accuracy.append(acc)
    print(f"Average Loss: {np.mean(losses)}")
    print(f"Average Accuracy: {np.mean(avg_accuracy)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="The model to use for testing", required=True
    )
    # print output?
    parser.add_argument(
        "-p", "--print", help="Print the output", action="store_true"
    )
    # batch size
    parser.add_argument(
        "-b", "--batch", help="Batch size", type=int, default=64
    )
    args = parser.parse_args()
    main(args)
    sys.exit(0)
