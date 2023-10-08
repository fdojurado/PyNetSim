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
from training import MixedDataModel, NetworkDataset, load_samples, get_all_samples, get_model
from pynetsim.utils import PyNetSimLogger
# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")
HIDDEN_SIZE = 512
NUM_EPOCHS = 100

logger_utility = PyNetSimLogger(log_file="my_log.log", namespace=__name__)
logger = logger_utility.get_logger()


def test_predicted_sample(y, output, print_output=False):
    # input(f"Y: {y}, shape: {y.shape}")
    # input(f"Shape of output: {output.shape}")
    _, predicted = torch.max(output.data, 1)
    # input(f"Predicted: {predicted}")
    correct = (predicted == y).sum().item()
    total = np.prod(y.shape)
    # input(f"Correct: {correct}, Total: {total}")
    if print_output:
        logger.info(f"Y: {y}")
        logger.info(f"Predicted: {predicted}")
        # get the index where the values are equal
        index = np.where(y == predicted)
        logger.info(f"Index: {index}")
        logger.info(f"Correct: {correct}, Total: {total}")
        input("Press enter to continue")
    return correct, total


def test_predicted_batch(y, output_batch, print_output=False):
    accuracy = []
    # input(f"Shape of output: {output_batch.shape}")
    # Loop through the batch [64, 100, 100]
    # We want to loop through the [1, 100, 100] and compare it with the y
    # for i in range(len(output_batch)):
        # input(f"Output: {output_batch[i]}, shape: {output_batch[i].shape}")
    correct, total = test_predicted_sample(
        y, output_batch, print_output)
    accuracy.append(correct/total * 100)
    return np.mean(accuracy)


def main(args):
    samples = load_samples(args.data)
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

    model, criterion, _ = get_model(load_model=args.model)

    model.eval()
    losses = []
    avg_accuracy = []
    with torch.no_grad():
        for input_data, categorical_data, target_data in test_loader:
            X = input_data
            # print(f"X: {X}, shape: {X.shape}")
            # multiply the first three columns by 6 and keep the rest the same
            # x_aux = X[:, :3] * 6
            # print(f"x_aux: {x_aux}, shape: {x_aux.shape}")
            # print(
            #     f"Categorical data: {categorical_data}, shape: {categorical_data.shape}")
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
