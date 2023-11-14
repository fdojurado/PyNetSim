import os
import json
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import pynetsim.leach.surrogate as leach_surrogate


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pynetsim.utils import PyNetSimLogger
from torch.optim import lr_scheduler
from rich.progress import Progress

# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")
PLOTS_PATH = os.path.join(SELF_PATH, "plots")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(namespace=__name__, log_file="my_log.log")
logger = logger_utility.get_logger()


class NetworkDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = x.shape[0]
        logger.info(f"X: {self.X.shape}, y: {self.y.shape}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # Support batching
    def collate_fn(self, batch):
        X = torch.stack([x[0] for x in batch])
        y = torch.stack([x[1] for x in batch])
        return X, y


class ClusterHeadModel(nn.Module):
    def __init__(self):
        super(ClusterHeadModel, self).__init__()

        self.lstm = nn.LSTM(input_size=35, hidden_size=64, num_layers=2, batch_first=True,
                            dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(208, 400)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128+400, 64)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 101*5)

    def forward(self, input_data):
        # print(f"Input: {x.shape}")
        # Get the none lstm features which are all the features except the last 35  features
        x = input_data[:, :-35]
        # print(f"None lstm features: {x.shape}")

        # Get the lstm features which are the last 35 features
        lstm_features = input_data[:, -35:]
        # print(f"lstm_features: {lstm_features.shape}")
        # [Batch, 35]

        # Pass the none lstm features through the first fully connected layer
        x = self.fc1(x)
        # print(f"fc1: {x.shape}")

        # Pass the output through the relu activation function
        x = self.relu(x)
        # print(f"relu: {x.shape}")

        # Pass the output through the dropout layer
        x = self.drop1(x)

        # Reshape the lstm features to [Batch, 5, 7]
        lstm_features = lstm_features.reshape(-1, 5, 7)
        # print(f"lstm_features: {lstm_features.shape}")

        # Get the lstm output
        lstm_out, _ = self.lstm(lstm_features)
        # print(f"lstm_out: {lstm_out.shape}")
        # [Batch, 128]
        # Get the last lstm output
        lstm_out = lstm_out[:, -1, :]
        # print(f"lstm_out: {lstm_out.shape}")

        # Concatenate the lstm output with the none lstm features
        x = torch.cat((x, lstm_out), dim=1)
        # print(f"Concatenated: {x.shape}")

        # Pass the concatenated features through the first fully connected layer
        x = self.fc2(x)
        # print(f"fc1: {x.shape}")
        # Pass the output through the relu activation function
        x = self.relu(x)
        # print(f"relu: {x.shape}")
        # Pass the output through the dropout layer
        x = self.drop2(x)
        # print(f"dropout: {x.shape}")
        # Pass the output through the second fully connected layer
        x = self.fc3(x)
        # print(f"fc2: {x.shape}")
        # Reshape the output to [Batch, 5, 101]
        x = x.reshape(-1, 5, 101)
        # print(f"Reshaped: {x.shape}")
        return x


class SurrogateModel:

    def __init__(self, config):
        self.name = "Cluster Head Regression Model"
        self.config = config
        self.epochs = self.config.surrogate.epochs
        self.hidden_dim = self.config.surrogate.hidden_dim
        self.output_dim = self.config.surrogate.output_dim
        self.num_clusters = self.config.surrogate.num_clusters
        self.weight_decay = self.config.surrogate.weight_decay
        self.drop_out = self.config.surrogate.drop_out
        self.batch_size = self.config.surrogate.batch_size
        self.learning_rate = self.config.surrogate.learning_rate
        self.test_ratio = self.config.surrogate.test_ratio
        self.largest_weight = self.config.surrogate.largest_weight
        self.largest_energy_level = self.config.surrogate.largest_energy_level
        self.max_dst_to_ch = self.config.surrogate.max_dst_to_ch
        self.num_workers = self.config.surrogate.num_workers
        self.load_model = self.config.surrogate.load_model
        self.model_path = self.config.surrogate.model_path
        self.raw_data_folder = self.config.surrogate.raw_data_folder
        self.data_folder = self.config.surrogate.data_folder
        self.plots_folder = self.config.surrogate.plots_folder
        self.print_every = self.config.surrogate.print_every
        self.plot_every = self.config.surrogate.plot_every
        self.eval_every = self.config.surrogate.eval_every

    def init(self):
        leach_surrogate.print_config(self.config, surrogate_name=self.name)

        # Create the folder to save the plots
        os.makedirs(self.plots_folder, exist_ok=True)

        # if data_path is not provided, then we need to generate the data
        if self.config.surrogate.generate_data:
            samples = leach_surrogate.generate_data(
                config=self.config)
        else:
            # Load the data
            samples = leach_surrogate.load_data(self.data_folder)

        logger.info(f"Number of samples: {len(samples)}")

        # Split the data into training and testing
        X_train, X_test, Y_train, Y_test = self.split_data(samples)

        # Create the training dataset
        self.training_dataset = NetworkDataset(x=X_train, y=Y_train)

        # Create the testing dataset
        self.testing_dataset = NetworkDataset(x=X_test, y=Y_test)

        # Create the training dataloader
        self.training_dataloader = DataLoader(
            self.training_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.training_dataset.collate_fn, num_workers=self.num_workers)

        # Create the testing dataloader
        self.testing_dataloader = DataLoader(
            self.testing_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

    def split_data(self, samples):
        # Get all the samples
        x, y, prev_x = self.get_all_samples(samples)
        # print shapes
        np_x = np.array(x)
        np_y = np.array(y)
        np_prev_x = np.array(prev_x)

        print(
            f"np_x: {np_x.shape}, np_y: {np_y.shape}, np_prev_x: {np_prev_x.shape}")

        # Concatenate the weights and current_membership
        np_x = np.concatenate(
            (np_x, np_prev_x), axis=1)

        # print y shape
        logger.info(f"np_y_chs: {np_y.shape}")

        np_y = np.eye(self.num_clusters+1)[np_y.astype(int)]
        print(f"np_y eye: {np_y.shape}")

        if self.test_ratio is None:
            raise Exception("Please provide the test ratio")

        # Lets split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            np_x, np_y, test_size=self.test_ratio, random_state=42, shuffle=True)

        # print shapes
        logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def get_all_samples(self, samples):
        x_data_list = []
        prev_x_data_list = []
        y_chs_list = []
        for _, sample in samples.items():
            for round in range(0, len(sample)):
                x_data = sample[str(round)]['x_data']
                prev_x_data = sample[str(round)]['prev_x_data']
                y_chs = sample[str(round)]['y_chs']
                x_data_list.append(x_data)
                prev_x_data_list.append(prev_x_data)
                y_chs_list.append(y_chs)

        return x_data_list, y_chs_list, prev_x_data_list

    def get_sample(self, samples, weights: tuple):
        x_data_list = []
        prev_x_data_list = []
        y_chs_list = []
        for key, sample in samples.items():
            if key == weights:
                for round in range(0, len(sample)):
                    x_data = sample[str(round)]['x_data']
                    prev_x_data = sample[str(round)]['prev_x_data']
                    y_chs = sample[str(round)]['y_chs']
                    x_data_list.append(x_data)
                    prev_x_data_list.append(prev_x_data)
                    y_chs_list.append(y_chs)

        return x_data_list, y_chs_list, prev_x_data_list

    def get_model(self, load_model=False):
        model = ClusterHeadModel()

        if self.load_model:
            if self.model_path is None:
                raise Exception(
                    "Please provide the path to the model to load")
            logger.info(f"Loading model: {self.model_path}")
            model.load_state_dict(torch.load(self.model_path))
        else:
            if load_model:
                raise Exception(
                    "Please provide the path to the model to load")
            logger.info(f"Creating new model")
            # Lets make sure that the folder to save the model exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        return model, criterion, optimizer, scheduler

    def train(self):
        if self.model_path is None:
            raise Exception("Please provide the path to save the model")

        model, criterion, optimizer, lr_scheduler = self.get_model()

        best_loss = float("inf")
        train_losses = []
        validation_losses = []

        for epoch in range(self.epochs):
            model.train()
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Training (epoch {epoch}/{self.epochs})", total=len(self.training_dataloader))
                for input_data, target_data in self.training_dataloader:
                    optimizer.zero_grad()
                    chs = model(
                        input_data=input_data)
                    loss = criterion(chs, target_data)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                    progress.update(task, advance=1)

            avg_train_loss = np.mean(train_losses)

            if epoch % self.print_every == 0:
                logger.info(
                    f"Epoch [{epoch}/{self.epochs}] Train Loss: {avg_train_loss:.4f}")

            if epoch % self.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for input_data, target_data in self.testing_dataloader:
                        chs = model(
                            input_data=input_data)
                        loss = criterion(chs, target_data)
                        validation_losses.append(loss.item())
                avg_val_loss = np.mean(validation_losses)
                if avg_val_loss < best_loss:
                    logger.info(
                        f"Epoch [{epoch}/{self.epochs}] Validation Loss Improved: {best_loss:.4f} -> {avg_val_loss:.4f}"
                    )
                    best_loss = avg_val_loss
                    if self.model_path:
                        torch.save(model.state_dict(), self.model_path)

            if epoch % self.plot_every == 0:
                plt.figure()  # Create a new figure
                plt.plot(train_losses, label="Train Loss")
                plt.plot(validation_losses, label="Validation Loss")
                plt.legend()
                plt.savefig(os.path.join(
                    self.plots_folder, f"train_validation_loss_classification.png"))
                plt.close()

            lr_scheduler.step()
            if epoch % 50 == 0:
                logger.info(
                    f"Updating the learning rate: {optimizer.param_groups[0]['lr']: .7f}")
        return model

    def test_predicted_sample(self, y, output, print_output=False):
        # Convert one hot encoded to categorical
        y = torch.argmax(y, dim=2)
        # _, predicted = torch.max(output.data, 1)
        # print(f"Predicted: {predicted}")
        _, predicted = torch.max(output.data, 2)
        # print(f"Predicted 1: {predicted}")
        correct = (predicted == y).sum().item()
        total = np.prod(y.shape)
        if print_output:
            logger.info(f"Y: {y}, chs: {np.unique(y)}")
            logger.info(f"Predicted: {predicted}, chs: {np.unique(predicted)}")
            # get the index where the values are equal
            index = np.where(y == predicted)
            logger.info(f"Correct index: {index}")
            # get the index where the values are not equal
            index = np.where(y != predicted)
            logger.info(f"Incorrect index: {index}")
            logger.info(f"Correct: {correct}, Total: {total}")
            input("Press enter to continue")
        return correct, total

    def test(self, batch: int = None, print_output=False, weights: list = None):
        logger.info(f"Testing with batch size: {batch}, weights: {weights}")
        # Lets check if the path to the model exists
        if self.model_path is None:
            raise Exception("Please provide the path to the model")

        model, criterion, _, _ = self.get_model(load_model=True)

        if weights is not None:
            if batch is not None:
                self.batch_size = batch
            # Convert weights to tuple
            weights = tuple(weights)
            # Load the data
            samples = leach_surrogate.load_data(self.data_folder)
            x, y = self.get_sample(
                samples, weights=weights)
            np_x = np.array(x)
            np_y = np.array(y)

            # Create target array with higher likelihoods for nodes in np_y_chs
            np_y_ext = np.zeros(
                (np_y.shape[0], self.num_clusters+1+5))
            for i in range(np_y.shape[0]):
                np_y_ext[i, np_y[i]] = 1

            self.testing_dataset = NetworkDataset(x=np_x, y=np_y_ext)
            # recreate the dataloader
            self.testing_dataloader = DataLoader(
                self.testing_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

        elif batch is not None:
            self.batch_size = batch
            # recreate the dataloader
            self.testing_dataloader = DataLoader(
                self.testing_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.testing_dataset.collate_fn, num_workers=self.num_workers)

        model.eval()
        losses = []
        avg_accuracy = []
        with torch.no_grad():
            for input_data, target_data in self.testing_dataloader:
                chs = model(input_data=input_data)
                loss = criterion(chs, target_data)
                losses.append(loss.item())
                correct, total = self.test_predicted_sample(
                    target_data, chs, print_output)
                acc = correct/total * 100
                avg_accuracy.append(acc)
        logger.info(f"Average Loss: {np.mean(losses)}")
        logger.info(f"Average Accuracy: {np.mean(avg_accuracy)}")
        logger.info(f"Accuracy Min: {np.min(avg_accuracy)}")
        logger.info(
            f"Number of samples with minimum accuracy: {np.sum(np.array(avg_accuracy) == np.min(avg_accuracy))}")
