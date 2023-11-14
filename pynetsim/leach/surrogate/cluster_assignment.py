import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pynetsim.leach.surrogate as leach_surrogate


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pynetsim.utils import PyNetSimLogger
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
        self.y = torch.from_numpy(y.astype(np.int64))
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


class ClusterAssignmentModel(nn.Module):
    def __init__(self):
        super(ClusterAssignmentModel, self).__init__()

        self.input_layer = nn.Linear(215, 400)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(400, 101)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


class ClusterAssignment:

    def __init__(self, config):
        self.name = "Cluster Assignment Model"
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
                func=self.process_data, config=self.config)
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
        x, y = self.get_all_samples(samples)
        # print shapes
        np_x = np.array(x)
        np_y = np.array(y)

        # print y shape
        print(f"np_y: {np_y.shape}")

        if self.test_ratio is None:
            raise Exception("Please provide the test ratio")

        # Lets split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            np_x, np_y, test_size=self.test_ratio, random_state=42, shuffle=True)

        # print shapes
        logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def get_round_data(self, name, stats, normalized_names_values: int, normalized_membership_values: int,
                       normalized_energy_values: int, normalized_dst_to_ch_values: int):
        energy_levels = list(stats['energy_levels'].values())
        energy_levels = [
            value / normalized_energy_values for value in energy_levels]
        assert all(-1 <= value <=
                   1 for value in energy_levels), f"Invalid energy levels: {energy_levels}"
        # membership = [0 if cluster_id is None else int(cluster_id) / normalized_membership_values
        #               for _, cluster_id in stats['membership'].items()]
        # # Remove the sink
        # membership = membership[1:]
        # assert all(
        #     0 <= value <= 1 for value in membership), f"Invalid membership: {membership}"

        # Get the remaining energy
        remaining_energy = stats['remaining_energy']/400
        assert 0 <= remaining_energy <= 1, f"Invalid remaining energy: {remaining_energy}"

        # Get distance to cluster head
        dst_to_cluster_head = [
            value / normalized_dst_to_ch_values for value in stats['dst_to_cluster_head'].values()]
        assert all(0 <= value <=
                   1 for value in dst_to_cluster_head), f"Invalid dst to cluster head: {dst_to_cluster_head}"

        # Get the alive nodes
        alive_nodes = stats['alive_nodes']/100
        assert 0 <= alive_nodes <= 1, f"Invalid alive nodes: {alive_nodes}"

        # Get the number of cluster heads
        num_cluster_heads = stats['num_cluster_heads']/5
        assert 0 <= num_cluster_heads <= 1, f"Invalid num cluster heads: {num_cluster_heads}"

        # Get the cluster heads
        if not stats['cluster_heads']:
            cluster_heads = [0] * 5
        else:
            cluster_heads = stats['cluster_heads']
            if len(cluster_heads) < 5:
                cluster_heads += [0] * (5-len(cluster_heads))

        # normalize the cluster heads
        cluster_heads = [
            value / normalized_membership_values for value in cluster_heads]
        cluster_heads.sort(reverse=False)

        assert len(
            cluster_heads) == 5 and all(0 <= value <= 1 for value in cluster_heads), f"Invalid cluster heads: {cluster_heads}"

        x_data = [value / normalized_names_values for value in name] + \
            energy_levels + \
            dst_to_cluster_head + \
            [remaining_energy, alive_nodes, num_cluster_heads] +\
            cluster_heads

        return x_data

    def process_data(self, samples, config):
        # Get keys of the dictionary
        all_keys = list(samples.keys())
        # Calculate the number of keys to keep (50% of the total keys)
        num_keys = int(len(all_keys) * 0.03)
        # Randomly select keys to keep
        keys_to_keep = random.sample(all_keys, num_keys)
        # Create a new dictionary with only the selected keys
        random_subset = {key: samples[key] for key in keys_to_keep}
        normalized_names_values = config.surrogate.largest_weight
        normalized_membership_values = 100
        normalized_energy_values = config.surrogate.largest_energy_level
        normalized_dst_to_ch_values = config.surrogate.max_dst_to_ch
        normalized_samples = {}
        for name, data in random_subset.items():
            normalized_samples[name] = []
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds-1:
                    continue

                rnd_data = self.get_round_data(
                    name, stats, normalized_names_values, normalized_membership_values, normalized_energy_values, normalized_dst_to_ch_values)

                next_round = round + 1

                # Get the cluster heads
                if not data[str(next_round)]['cluster_heads']:
                    next_round_chs = [0] * 5
                else:
                    next_round_chs = data[str(next_round)]['cluster_heads']
                    if len(next_round_chs) < 5:
                        next_round_chs += [0] * (5-len(next_round_chs))
                # normalize the cluster heads
                next_round_chs = [
                    value / normalized_membership_values for value in next_round_chs]
                next_round_chs.sort(reverse=False)

                assert len(
                    next_round_chs) == 5 and all(0 <= value <= 1 for value in next_round_chs), f"Invalid next round chs: {next_round_chs}"

                # Add the cluster heads to the round data
                rnd_data += next_round_chs

                assert all(-1 <= value <=
                           1 for value in rnd_data), f"Invalid rnd_data: {rnd_data}"

                next_round_membership = [0 if cluster_id is None else int(
                    cluster_id) for _, cluster_id in data[str(next_round)]['membership'].items()]

                # remove the sink
                y = next_round_membership[1:]

                assert len(y) == 99, f"Invalid y: {y}"

                for i in range(2, 100):
                    # add i to rnd_data
                    rnd_data_chs = rnd_data + [i/100]
                    # Select next round membership
                    nxt_rnd_memb = y[i-2]
                    assert all(-1 <= value <=
                               1 for value in rnd_data_chs), f"Invalid rnd_data_chs: {rnd_data_chs}"
                    assert 0 <= nxt_rnd_memb <= 100, f"Invalid nxt_rnd_memb: {nxt_rnd_memb}"
                    normalized_samples[name].append(
                        {'x': rnd_data_chs, 'y': nxt_rnd_memb})

        os.makedirs(self.data_folder, exist_ok=True)
        for name, data in normalized_samples.items():
            with open(os.path.join(self.data_folder, f"{name}.json"), "w") as f:
                json.dump(data, f)

        return normalized_samples

    def get_all_samples(self, samples):
        x_data_list = []
        y_list = []
        for _, sample in samples.items():
            for i in range(0, len(sample)):
                x_data = sample[i]['x']
                y_chs = sample[i]['y']
                x_data_list.append(x_data)
                y_list.append(y_chs)

        return x_data_list, y_list

    def get_sample(self, samples, weights: tuple):
        x_data_list = []
        y_list = []
        for key, sample in samples.items():
            if key == weights:
                for i in range(0, len(sample)):
                    x_data = sample[i]['x']
                    y_chs = sample[i]['y']
                    x_data_list.append(x_data)
                    y_list.append(y_chs)

        return x_data_list, y_list

    def get_model(self, load_model=False):
        model = ClusterAssignmentModel()

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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate)

        return model, criterion, optimizer

    def train(self):
        if self.model_path is None:
            raise Exception("Please provide the path to save the model")

        model, criterion, optimizer = self.get_model()

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
                        x=input_data)
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
                            x=input_data)
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

        return model

    def test_predicted_sample(self, y, output, print_output=False):
        predicted = torch.argmax(output, dim=1)
        # How many y values are in the predicted
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

        model, criterion, _ = self.get_model(load_model=True)

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
            np_y_ext = np.eye(self.num_clusters+1)[np_y]

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
                chs = model(x=input_data)
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
