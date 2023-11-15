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
from tqdm import tqdm

# Constants
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.dirname(SELF_PATH)
RESULTS_PATH = os.path.join(TUTORIALS_PATH, "results")
MODELS_PATH = os.path.join(SELF_PATH, "models")
PLOTS_PATH = os.path.join(SELF_PATH, "plots")

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(namespace=__name__, log_file="my_log.log")
logger = logger_utility.get_logger()


class ClusterHeadDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # Support batching
    def collate_fn(self, batch):
        X = torch.stack([x[0] for x in batch])
        y = torch.stack([x[1] for x in batch])
        return X, y


class ForecastCCH(nn.Module):
    def __init__(self):
        super(ForecastCCH, self).__init__()
        self.batch_norm = nn.BatchNorm1d(1535)
        self.lstm = nn.LSTM(input_size=307, hidden_size=700, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(1400, 101*5)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        # [batch_size, 1535]

        # Apply batch normalization
        x = self.batch_norm(x)
        # reshape input to be [samples, time steps, features]
        x = x.view(-1, 5, int(x.shape[1]/5))
        # print(f"Reshaped input shape: {x.shape}")
        # [batch_size, 5, 307]

        # Forward propagate LSTM
        out, _ = self.lstm(x)
        # print(f"Output shape: {out.shape}")
        # [batch_size, 5, 1000]

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # print(f"Output shape: {out.shape}")
        # [batch_size, 101*5]

        # reshape output to be [samples, 5, 101]
        out = out.view(-1, 5, 101)
        # print(f"Reshaped output shape: {out.shape}")
        # [batch_size, 5, 101]

        return out


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

        logger.info(f"Data shape: {samples.shape}")

        training_set, validation_set = leach_surrogate.get_data(
            samples, train_fraction=0, test_fraction=0.2)

        logger.info(
            f"Proportion of training set: {len(training_set)/len(samples)*100:.2f}%")
        logger.info(
            f"Proportion of testing set: {len(validation_set)/len(samples)*100:.2f}%")

        n_steps = 5
        x_train, y_train = self.split_sequence(training_set, n_steps)
        y_train = np.eye(101)[y_train.astype('int')]
        x_val, y_val = self.split_sequence(validation_set, n_steps)
        y_val = np.eye(101)[y_val.astype('int')]

        print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")

        # Create the training dataset
        self.training_dataset = ClusterHeadDataset(x_train, y_train)

        # Create the testing dataset
        self.testing_dataset = ClusterHeadDataset(x_val, y_val)

        if x_train.shape[0] > 0:
            # Create the training dataloader
            self.train_loader = DataLoader(
                self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            self.train_loader = None

        if x_val.shape[0] > 0:
            # Create the testing dataloader
            self.valid_loader = DataLoader(
                self.testing_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            self.valid_loader = None

    def split_sequence(self, sequence, n_steps):
        x_data = []
        y_data = []
        num_samples = len(sequence)

        for i in tqdm(range(num_samples), desc="Processing sequence"):
            end_ix = i + n_steps
            if end_ix > num_samples - 1:
                break
            alpha_val, beta_val, gamma_val = sequence['alpha'][i:end_ix].values, sequence['beta'][
                i:end_ix].values, sequence['gamma'][i:end_ix].values
            # convert to float
            alpha_val = [float(x)/10 for x in alpha_val]
            beta_val = [float(x)/10 for x in beta_val]
            gamma_val = [float(x)/10 for x in gamma_val]
            assert all(
                x <= 1 and x >= -1 for x in alpha_val), f"Incorrect values of alpha: {alpha_val}"
            assert all(
                x <= 1 and x >= -1 for x in beta_val), f"Incorrect values of beta: {beta_val}"
            assert all(
                x <= 1 and x >= -1 for x in gamma_val), f"Incorrect values of gamma: {gamma_val}"
            # Normalize remaining energy dividing by 10
            remaining_energy = sequence['remaining_energy'][i:end_ix]
            remaining_energy = [float(x)/10 for x in remaining_energy]
            assert all(
                x <= 1 and x >= -1 for x in remaining_energy), f"Incorrect values of remaining energy: {remaining_energy}"
            # seq_x.extend(remaining_energy)
            # Normalize alive nodes dividing by 100
            alive_nodes = sequence['alive_nodes'][i:end_ix].values
            alive_nodes = [float(x)/100 for x in alive_nodes]
            assert all(
                x <= 1 and x >= -1 for x in alive_nodes), f"Incorrect values of alive nodes: {alive_nodes}"
            # seq_x.extend(alive_nodes)
            # Normalize energy levels dividing by 5
            energy_levels = sequence['energy_levels'][i:end_ix].values
            energy_levels = [eval(x) for x in energy_levels]
            # Convert to float every element in the array of arrays
            energy_levels = [[float(x)/5 for x in sublist]
                             for sublist in energy_levels]
            # energy levels is a list of lists, so we need to assert that all values are between -1 and 1
            # We iterate over the list of lists and assert that all values are between -1 and 1
            assert all(
                -1 <= x <= 1 for sublist in energy_levels for x in sublist), f"Incorrect values of energy levels: {energy_levels}"
            # seq_x.extend(energy_levels)
            # Normalize distance to cluster head dividing by 100
            dst_to_cluster_head = sequence['dst_to_cluster_head'][i:end_ix].values
            dst_to_cluster_head = [eval(x) for x in dst_to_cluster_head]
            dst_to_cluster_head = [[float(x)/200 for x in sublist]
                                   for sublist in dst_to_cluster_head]
            assert all(-1 <= x <=
                       1 for sublist in dst_to_cluster_head for x in sublist), f"Incorrect values of distance to cluster head: {dst_to_cluster_head}"

            # seq_x.extend(dst_to_cluster_head)
            # Normalize membership dividing by 100
            membership = sequence['membership'][i:end_ix].values
            membership = [eval(x) for x in membership]
            membership = [[float(x)/100 for x in sublist]
                          for sublist in membership]
            assert all(-1 <= x <=
                       1 for sublist in membership for x in sublist), f"Incorrect values of membership: {membership}"
            # seq_x.extend(membership)
            # Normalize cluster heads dividing by 100
            chs, seq_y = sequence['cluster_heads'][i:
                                                   end_ix], sequence['cluster_heads'][end_ix]
            chs = [eval(x) for x in chs]
            chs = [[float(x)/100 for x in sublist] for sublist in chs]
            assert all(-1 <= x <=
                       1 for sublist in chs for x in sublist), f"Incorrect values of cluster heads: {chs}"

            seq_y = eval(seq_y)

            next_alpha_val, next_beta_val, next_gamma_val = sequence['alpha'][end_ix], sequence['beta'][
                end_ix], sequence['gamma'][end_ix]
            # convert to float
            next_alpha_val = float(next_alpha_val)/10
            next_beta_val = float(next_beta_val)/10
            next_gamma_val = float(next_gamma_val)/10

            if (next_alpha_val != alpha_val[0]) or (next_beta_val != beta_val[0]) or (next_gamma_val != gamma_val[0]):
                continue

            # Lets put the data into the seq_x like this weights[0], remaining energy[0],...,weights[1], remaining energy[1]
            seq_x_tmp = []
            for i in range(n_steps):
                a = alpha_val[i]
                b = beta_val[i]
                g = gamma_val[i]
                re = remaining_energy[i]
                an = alive_nodes[i]
                el = energy_levels[i]
                dst = dst_to_cluster_head[i]
                mem = membership[i]
                ch = chs[i]
                # Put the alpha, beta, gamma, remaining energy and alive nodes at the end of the list
                exp = []
                # seq_x.extend([a, b, g, re, an])
                exp.extend([a, b, g, re, an])
                exp.extend(el)
                exp.extend(dst)
                exp.extend(mem)
                exp.extend(ch)
                # Append to the list
                seq_x_tmp.append(exp)
                # seq_x.extend([a, b, g, re, an, el, dst, mem, ch])

            # Convert seq_x into a single list
            seq_x = [item for sublist in seq_x_tmp for item in sublist]

            x_data.append(seq_x)
            y_data.append(seq_y)

        return np.array(x_data), np.array(y_data)

    def get_model(self, load_model=False):
        model = ForecastCCH()

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
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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
                for input_data, target_data in self.train_loader:
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
                    for input_data, target_data in self.valid_loader:
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
                chs = model(input_data)
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
