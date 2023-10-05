import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")

# Configuration parameters
BATCH_SIZE = 64
INPUT_SIZE = 203
HIDDEN_SIZE = 512
OUTPUT_SIZE = 99
LEARNING_RATE = 1e-6
NUM_EPOCHS = 5000
LARGEST_WEIGHT = 6
NUM_CLUSTERS = 100

# Print and plot intervals
PRINT_EVERY = 1
PLOT_EVERY = 10


class NetworkDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y


class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassificationModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


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
                          for node_id, cluster_id in stats['membership'].items()
                          if node_id not in {1, '1'}]

            x_data = [value / normalized_names_values for value in name] + \
                energy_levels + membership

            next_round = round + 1
            next_round_membership = [0 if cluster_id is None else int(
                cluster_id) / normalized_membership_values for node_id, cluster_id in data[str(next_round)]['membership'].items() if node_id not in {1, '1'}]

            y_data = next_round_membership

            assert all(-1 <= value <=
                       1 for value in x_data), f"Invalid x_data: {x_data}"
            assert all(
                0 <= value <= 1 for value in y_data), f"Invalid y_data: {y_data}"

            normalized_samples[name][round] = {
                "x_data": x_data,
                "y_data": y_data
            }

    os.makedirs("data_classification", exist_ok=True)
    for name, data in normalized_samples.items():
        with open(os.path.join("data_classification", f"{name}.json"), "w") as f:
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
            # input(f"Name: {name}")

            with open(os.path.join(data_dir, file), "r") as f:
                data = json.load(f)
            samples[name] = data
    return samples


def load_samples(data_dir):
    print(f"Loading samples from: {data_dir}")
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


def train_model(load_model, train_loader, test_loader, input_size, hidden_size, output_size, num_epochs, learning_rate, model_path=None):
    model = ClassificationModel(input_size, hidden_size, output_size)

    if load_model:
        print(f"Loading model: {load_model}")
        model.load_state_dict(torch.load(load_model))

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        for input_data, target_data in train_loader:
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = criterion(outputs, target_data)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for input_data, target_data in test_loader:
                outputs = model(input_data)
                loss = criterion(outputs, target_data)
                validation_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(validation_losses)

        if epoch % PRINT_EVERY == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}")

        if epoch % PLOT_EVERY == 0:
            plt.figure()  # Create a new figure
            plt.plot(train_losses, label="Train Loss")
            plt.plot(validation_losses, label="Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(
                "plots", f"train_validation_loss_classification.png"))

        if avg_val_loss < best_loss:
            print(
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
    for key, sample in samples.items():
        for round in range(1, len(sample)+1):
            x_data = sample[str(round)]['x_data']
            y_data = sample[str(round)]['y_data']
            x.append(x_data)
            y.append(y_data)

    return x, y


def main(args):

    # Create a folder to save the model
    os.makedirs("models", exist_ok=True)

    # Plot folder
    os.makedirs("plots", exist_ok=True)

    if args.data is None:
        files = load_files(RESULTS_PATH)
        samples = normalize_data(files)
    else:
        # Load the training and testing data
        samples = load_samples(args.data)

    x, y = get_all_samples(samples)

    y = np.array(y)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    x = np.array(x)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=False)

    train_dataset = NetworkDataset(X_train, y_train)
    test_dataset = NetworkDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=test_dataset.collate_fn)

    model_path = os.path.join(MODELS_PATH, "classification_model.pt")

    train_model(args.load, train_loader, test_loader, train_dataset.x.shape[1],
                HIDDEN_SIZE, train_dataset.y.shape[1], NUM_EPOCHS, LEARNING_RATE, model_path)

    print("Training completed.")
    sys.exit(0)


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
