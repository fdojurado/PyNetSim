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
from pynetsim.utils import PyNetSimLogger
from training_classification import NetworkDataset, load_samples, get_all_samples, get_model

# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")
HIDDEN_SIZE = 512
NUM_EPOCHS = 100

logger_utility = PyNetSimLogger(log_file="my_log.log", namespace=__name__)
logger = logger_utility.get_logger()


def closest_value(output):
    return (output * 100).round().astype(int)


def test_predicted_sample(encoder, x, y, output, print_output=False):
    y = encoder.inverse_transform(y)
    output = encoder.inverse_transform(output)
    y = closest_value(y)
    output = closest_value(output)
    correct = np.sum(y == output)
    total = np.prod(y.shape)
    if print_output:
        logger.info(f"Y: {y}")
        logger.info(f"predicted: {output}")
        # Get the index where the values are equal
        index = np.where(y == output)
        logger.info(f"Index: {index}")
        logger.info(f"Correct: {correct}, Total: {total}")
        input("Press Enter to continue...")
    return correct, total


def test_predicted_batch(encoder, x, y, output, print_output=False):
    accuracy = []
    for i in range(len(output)):
        ith_output = output[i].reshape(1, -1)
        ith_y = y[i].reshape(1, -1)
        ith_x = x[i]
        correct, total = test_predicted_sample(
            encoder, ith_x, ith_y, ith_output, print_output)
        accuracy.append(correct/total * 100)
    return np.mean(accuracy)


def main(args):
    samples = load_samples(args.data)
    x, y = get_all_samples(samples)
    y = np.array(y)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    x = np.array(x)
    _, X_test, _, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=False)
    test_dataset = NetworkDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collate_fn)

    model, criterion, _ = get_model(test_dataset.x.shape[1],
                                    HIDDEN_SIZE, test_dataset.y.shape[1], 0, args.model)

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
            if args.batch == 1:
                correct, total = test_predicted_sample(
                    encoder, X, y, output, args.print)
                avg_accuracy.append(correct/total*100)
                continue
            acc = test_predicted_batch(encoder, X, y, output, args.print)
            avg_accuracy.append(acc)
    logger.info(f"Average Loss: {np.mean(losses)}")
    logger.info(f"Average Accuracy: {np.mean(avg_accuracy)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", help="Path to the training and testing folder", default=None)
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
