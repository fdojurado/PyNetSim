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
BATCH_SIZE = 64
HIDDEN_SIZE = 512
NUM_EPOCHS = 100


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


def load_samples():
    samples = {}
    data_dir = os.path.join(SELF_PATH, "data_classification")
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
    return (output * 100).round().astype(int)


def test_predicted_sample(encoder, x, y, output):
    y = encoder.inverse_transform(y)
    output = encoder.inverse_transform(output)
    y = closest_value(y)
    output = closest_value(output)
    correct = np.sum(y == output)
    total = np.prod(y.shape)
    # Index where the values are equal
    # index = np.where(y == output)
    # print(f"Y: {y}")
    # print(f"Output: {output}")
    # print(f"Index: {index}")
    # input(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total*100}")
    return correct, total


def test_predicted_batch(encoder, x, y, output):
    accuracy = []
    for i in range(len(output)):
        ith_output = output[i].reshape(1, -1)
        ith_y = y[i].reshape(1, -1)
        ith_x = x[i]
        correct, total = test_predicted_sample(
            encoder, ith_x, ith_y, ith_output)
        accuracy.append(correct/total * 100)
    return np.mean(accuracy)


def main(args):
    print(f"Loading model: {args.model}")
    samples = load_samples()
    x, y = get_all_samples(samples)
    y = np.array(y)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    x = np.array(x)
    _, X_test, _, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=False)
    test_dataset = NetworkDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_dataset.collate_fn)
    model = ClassificationModel(
        test_dataset.x.shape[1], HIDDEN_SIZE, test_dataset.y.shape[1])
    criterion = nn.BCELoss()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    losses = []
    avg_accuracy = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            X = batch_x
            y = batch_y
            output = model(X)
            loss = criterion(output, y)
            losses.append(loss.item())
            if BATCH_SIZE == 1:
                correct, total = test_predicted_sample(encoder, X, y, output)
                avg_accuracy.append(correct/total*100)
                continue
            acc = test_predicted_batch(encoder, X, y, output)
            avg_accuracy.append(acc)
    print(f"Average Loss: {np.mean(losses)}")
    print(f"Average Accuracy: {np.mean(avg_accuracy)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="The model to use for testing", required=True
    )
    args = parser.parse_args()
    main(args)
    sys.exit(0)
