#     PyNetSim: A Python-based Network Simulator for Low-Energy Adaptive Clustering Hierarchy (LEACH) Protocol
#     Copyright (C) 2024  F. Fernando Jurado-Lasso (ffjla@dtu.dk)

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = x.shape[0]
        print(f"len: {self.len}")
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


class MixedDataModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_labels, drop_out):
        super(MixedDataModel, self).__init__()
        self.num_classes = num_classes
        self.num_labels = num_labels

        # Cluster heads predictions sequential model
        self.ch_layer = nn.Sequential(
            nn.Linear(304, 500),
            nn.ReLU(),
            nn.Linear(500, 101),
            nn.Softmax(dim=1)
        )

        # Cluster assignment sequential model
        self.ch_assignment_layer = nn.Sequential(
            nn.Linear(309, 500),
            nn.ReLU(),
            nn.Linear(500, 101),
            nn.Softmax(dim=1)
        )

    def forward(self, x, next_chs: list = None):

        input_data = x
        # input_data = input_data.unsqueeze(1)
        # print(f"Input data shape: {input_data.shape}")
        # [Batch, 303]

        # Get the membership which are the last 99 elements
        # membership = input_data[:, -99:]
        # print(f"Membership shape: {membership.shape}")
        # [Batch, 99]

        # Get the first elements but not the last 99 elements
        # input_data = input_data[:, :-99]
        # print(f"Input data shape: {input_data.shape}")
        # [Batch, 204]

        # Expand the input_data to [Batch, 5, 205], where in the last dimension we add a number from 1 to 5
        ch_input_data = input_data.unsqueeze(1).expand(
            (-1, 5, input_data.shape[1]))
        # print(f"ch_input_data shape: {ch_input_data.shape}")
        # [Batch, 5, 303]

        # Now lets add the cluster head number so the input data has shape [Batch, 5, 205]
        cluster_head_number = torch.arange(
            1, 6).unsqueeze(0).expand((ch_input_data.shape[0], 5)).unsqueeze(2)
        # print(f"cluster_head_number shape: {cluster_head_number.shape}")
        # [Batch, 5, 1]

        # Normalize the cluster_head_number by dividing by 5
        cluster_head_number = cluster_head_number / 5

        # Lets concatenate the cluster_head_number to the ch_input_data
        ch_input_data = torch.cat((ch_input_data, cluster_head_number), dim=2)
        # print(f"ch_input_data shape: {ch_input_data.shape}")
        # [Batch, 5, 304]

        # Lets print the ch_input_data
        # print(f"ch_input_data: {ch_input_data}, {ch_input_data.shape}")

        # Pass the input data through the cluster heads prediction layer
        ch_output = self.ch_layer(ch_input_data)
        # print(f"ch_output: {ch_output}, {ch_output.shape}")

        # print(f"membership: {membership}, {membership.shape}")
        if next_chs is not None:
            # Lets concatenate the next_chs
            next_chs = torch.argmax(next_chs, dim=2).float()
            next_chs = next_chs / 100
            # print(f"next_chs: {next_chs}, {next_chs.shape}")
            ch_assign_input = torch.cat((next_chs, x), dim=1)
            # Expand the ch_assign_input to [Batch, 99, 308], where in the last dimension we add a number from 1 to 99
            ch_assign_input = ch_assign_input.unsqueeze(1).expand(
                (-1, 99, ch_assign_input.shape[1]))
            # print(f"ch_assign_input shape: {ch_assign_input.shape}")
            # [Batch, 99, 308]

            # Now lets add the Node ID so the input data has shape [Batch, 99, 309]
            node_id = torch.arange(
                1, 100).unsqueeze(0).expand((ch_assign_input.shape[0], 99)).unsqueeze(2)
            # print(f"node_id shape: {node_id.shape}")
            # [Batch, 99, 1]

            # Normalize the node_id by dividing by 99
            node_id = node_id / 99

            # Lets concatenate the node_id to the ch_assign_input
            ch_assign_input = torch.cat(
                (ch_assign_input, node_id), dim=2)
            # [Batch, 99, 309]
            # print(f"ch assignment shape: {ch_assign_input.shape}")
        else:
            # Argmax the ch_output
            ch_output_arg_max = torch.argmax(ch_output, dim=2).float()
            print(
                f"ch_output argmax: {ch_output_arg_max}, {ch_output_arg_max.shape}")
            # Normalize the ch_output_arg_max by dividing by the number of classes
            ch_output_arg_max = ch_output_arg_max / 100
            ch_assign_input = torch.cat(
                (ch_output_arg_max, x), dim=1)
            # Expand the ch_assign_input to [Batch, 99, 308], where in the last dimension we add a number from 1 to 99
            ch_assign_input = ch_assign_input.unsqueeze(1).expand(
                (-1, 99, ch_assign_input.shape[1]))
            print(f"ch assignment  shape: {ch_assign_input.shape}")
            # [Batch, 99, 308]

            # Now lets add the cluster head number so the input data has shape [Batch, 99, 309]
            node_id = torch.arange(
                1, 100).unsqueeze(0).expand((ch_assign_input.shape[0], 99)).unsqueeze(2)
            print(f"ch assignment  shape: {node_id.shape}")
            # [Batch, 99, 1]

            # Normalize the node_id by dividing by 99
            node_id = node_id / 99

            # Lets concatenate the cluster_head_number to the ch_assign_input
            ch_assign_input = torch.cat(
                (ch_assign_input, node_id), dim=2)
            # [Batch, 99, 309]
            print(f"ch assignment shape: {ch_assign_input.shape}")

        # Pass the ch_assignment_input through the ch_assignment_layer
        ch_assignment_output = self.ch_assignment_layer(ch_assign_input)
        # print(
        #     f"ch_assignment_output: {ch_assignment_output}, {ch_assignment_output.shape}")

        return ch_output, ch_assignment_output


class SurrogateModel:

    def __init__(self, config):
        self.config = config
        self.lstm_arch = config.surrogate.lstm_arch
        self.epochs = self.config.surrogate.epochs
        self.hidden_dim = self.config.surrogate.hidden_dim
        self.lstm_hidden = self.config.surrogate.lstm_hidden
        self.output_dim = self.config.surrogate.output_dim
        self.num_clusters = self.config.surrogate.num_clusters
        self.num_embeddings = self.config.surrogate.num_embeddings
        self.embedding_dim = self.config.surrogate.embedding_dim
        self.numeral_dim = self.config.surrogate.numerical_dim
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

    def print_config(self):
        # print all the config values
        logger.info(f"lstm_arch: {self.lstm_arch}")
        logger.info(f"epochs: {self.epochs}")
        logger.info(f"hidden_dim: {self.hidden_dim}")
        logger.info(f"lstm_hidden: {self.lstm_hidden}")
        logger.info(f"output_dim: {self.output_dim}")
        logger.info(f"num_clusters: {self.num_clusters}")
        logger.info(f"num_embeddings: {self.num_embeddings}")
        logger.info(f"embedding_dim: {self.embedding_dim}")
        logger.info(f"numeral_dim: {self.numeral_dim}")
        logger.info(f"weight_decay: {self.weight_decay}")
        logger.info(f"drop_out: {self.drop_out}")
        logger.info(f"batch_size: {self.batch_size}")
        logger.info(f"learning_rate: {self.learning_rate}")
        logger.info(f"test_ratio: {self.test_ratio}")
        logger.info(f"largest_weight: {self.largest_weight}")
        logger.info(f"largest_energy_level: {self.largest_energy_level}")
        logger.info(f"num_workers: {self.num_workers}")
        logger.info(f"load_model: {self.load_model}")
        logger.info(f"model_path: {self.model_path}")
        logger.info(f"raw_data_folder: {self.raw_data_folder}")
        logger.info(f"data_folder: {self.data_folder}")
        logger.info(f"plots_folder: {self.plots_folder}")
        logger.info(f"print_every: {self.print_every}")
        logger.info(f"plot_every: {self.plot_every}")
        logger.info(f"eval_every: {self.eval_every}")

    def init(self):
        self.print_config()
        # Create the folder to save the model if it does not exist
        # os.makedirs(self.model_path, exist_ok=True)

        # Create the folder to save the plots
        os.makedirs(self.plots_folder, exist_ok=True)

        # if data_path is not provided, then we need to generate the data
        if self.config.surrogate.generate_data:
            samples = self.generate_data()
        else:
            # Load the data
            samples = self.load_data()

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
        x, y, prev_x, y_chs = self.get_all_samples(samples)
        # print shapes
        np_x = np.array(x)
        np_y = np.array(y)
        np_prev_x = np.array(prev_x)
        np_y_chs = np.array(y_chs)

        # Lets one hot encode the y
        # print(f"np_y: {np_y[0]}")
        # np_y = np.eye(self.num_clusters+1)[np_y.astype(int)]

        # Concatenate the weights and current_membership
        # np_x = np.concatenate(
        #     (np_x, np_prev_x), axis=1)
        # print y shapes
        print(f"np_y: {np_y.shape}")
        print(f"np_y_chs: {np_y_chs.shape}")
        # Concatenate the y and y_chs
        np_y = np.concatenate(
            (np_y, np_y_chs), axis=1)

        print(f"np_y: {np_y.shape}")
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

    def generate_data(self):
        if self.raw_data_folder is None:
            raise Exception(
                "Please provide the path to the raw data folder to generate the data")
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to save the generated data")
        # Load the data folder
        files = self.load_files(self.raw_data_folder)
        # Normalize the data
        # samples = self.normalize_data(files, normalized_names_values=self.largest_weight,
        #                               normalized_membership_values=1)
        samples = self.normalize_data_cluster_heads(
            files, normalized_names_values=self.largest_weight, normalized_membership_values=100,
            normalized_energy_values=self.largest_energy_level, normalized_dst_to_ch_values=self.max_dst_to_ch)

        return samples

    def load_data(self):
        if self.data_folder is None:
            raise Exception(
                "Please provide the path to the data folder to load the data")
        # Load the data folder
        samples = self.load_samples(self.data_folder)

        return samples

    def get_round_data(self, name, stats, normalized_names_values: int, normalized_membership_values: int,
                       normalized_energy_values: int, normalized_dst_to_ch_values: int):
        energy_levels = list(stats['energy_levels'].values())
        energy_levels = [
            value / normalized_energy_values for value in energy_levels]
        assert all(-1 <= value <=
                   1 for value in energy_levels), f"Invalid energy levels: {energy_levels}"
        membership = [0 if cluster_id is None else int(cluster_id) / normalized_membership_values
                      for _, cluster_id in stats['membership'].items()]
        # Remove the sink
        membership = membership[1:]
        assert all(
            0 <= value <= 1 for value in membership), f"Invalid membership: {membership}"

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

        # Get control packets energy
        # control_packets_energy = stats['control_packets_energy']/10
        # assert 0 <= control_packets_energy <= 1, f"Invalid control packets energy: {control_packets_energy}"

        # Get control packets bits
        # control_pkt_bits = stats['control_pkt_bits']/1e8
        # assert 0 <= control_pkt_bits <= 1, f"Invalid control packets bits: {control_pkt_bits}"

        # Get the pkts sent to bs
        # pkts_sent_to_bs = stats['pkts_sent_to_bs']/1e3
        # assert 0 <= pkts_sent_to_bs <= 1, f"Invalid pkts sent to bs: {pkts_sent_to_bs}"

        # Get the pkts received by bs
        # pkts_recv_by_bs = stats['pkts_recv_by_bs']/1e3
        # assert 0 <= pkts_recv_by_bs <= 1, f"Invalid pkts recv by bs: {pkts_recv_by_bs}"

        # Get the energy dissipated
        # energy_dissipated = stats['energy_dissipated']/200
        # assert 0 <= energy_dissipated <= 1, f"Invalid energy dissipated: {energy_dissipated}"

        x_data = [value / normalized_names_values for value in name] + \
            energy_levels + \
            dst_to_cluster_head + \
            [remaining_energy, alive_nodes, num_cluster_heads] + \
            membership
        #  control_packets_energy, control_pkt_bits, pkts_sent_to_bs, pkts_recv_by_bs, energy_dissipated] + \

        return x_data

    def normalize_data_cluster_heads(self, samples, normalized_names_values: int, normalized_membership_values: int,
                                     normalized_energy_values: int, normalized_dst_to_ch_values: int):
        normalized_samples = {}
        for name, data in samples.items():
            normalized_samples[name] = {}
            max_rounds = len(data)

            for round, stats in data.items():
                round = int(round)
                if round == max_rounds-1:
                    continue

                rnd_data = self.get_round_data(
                    name, stats, normalized_names_values, normalized_membership_values, normalized_energy_values, normalized_dst_to_ch_values)

                # Lets attached 10 past experiences
                prev_x_data = []
                prev_round = round - 1
                # for prev_round in range(round-1, round-11, -1):
                if prev_round < 0:
                    prev_round_data = [1] * 99
                    prev_round_data += [1] * 99
                    prev_round_data += [1]
                    prev_round_data += [1]
                    prev_round_data += [1]
                    prev_round_data += [1] * 99

                else:
                    prev_round_data = normalized_samples[name][str(
                        prev_round)]['x_data']
                    # Remove the first 3 elements
                    prev_round_data = prev_round_data[3:]
                prev_x_data = [0]

                next_round = round + 1
                next_round_membership = [0 if cluster_id is None else int(
                    cluster_id) for _, cluster_id in data[str(next_round)]['membership'].items()]

                # Get the cluster heads
                if not data[str(next_round)]['cluster_heads']:
                    next_round_chs = [0] * 5
                else:
                    next_round_chs = data[str(next_round)]['cluster_heads']
                    if len(next_round_chs) < 5:
                        next_round_chs += [0] * (5-len(next_round_chs))

                assert len(
                    next_round_chs) == 5, f"Invalid chs: {next_round_chs}"

                # Remove the sink
                next_round_membership = next_round_membership[1:]

                y_data = next_round_membership

                assert len(y_data) == 99, f"Invalid y_data: {y_data}"

                # assert all(-1 <= value <=
                #            1 for value in rnd_data), f"Invalid x_data: {rnd_data}"
                assert all(
                    0 <= value <= self.num_clusters for value in y_data), f"Invalid y_data: {y_data}"

                normalized_samples[name][str(round)] = {
                    "x_data": rnd_data,
                    "prev_x_data": prev_x_data,
                    "y_data": y_data,
                    "y_chs": next_round_chs,
                }

        os.makedirs(self.data_folder, exist_ok=True)
        for name, data in normalized_samples.items():
            with open(os.path.join(self.data_folder, f"{name}.json"), "w") as f:
                json.dump(data, f)

        return normalized_samples

    def normalize_data(self, samples, normalized_names_values: int, normalized_membership_values: int):
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
                              for node_id, cluster_id in stats['membership'].items()]

                # Add the sink at the beginning
                membership.insert(0, 0)

                x_data = [value / normalized_names_values for value in name] + \
                    energy_levels

                next_round = round + 1
                next_round_membership = [0 if cluster_id is None else int(
                    cluster_id) / normalized_membership_values for node_id, cluster_id in data[str(next_round)]['membership'].items()]

                # Add the sink at the beginning
                next_round_membership.insert(0, 0)

                y_data = next_round_membership

                assert all(-1 <= value <=
                           1 for value in x_data), f"Invalid x_data: {x_data}"
                assert all(
                    0 <= value <= self.num_clusters for value in y_data), f"Invalid y_data: {y_data}"
                assert all(
                    0 <= value <= self.num_clusters for value in membership), f"Invalid membership: {membership}"

                normalized_samples[name][str(round)] = {
                    "x_data": x_data,
                    "y_data": y_data,
                    "membership": membership
                }

        os.makedirs(self.data_folder, exist_ok=True)
        for name, data in normalized_samples.items():
            # with open(os.path.join("data", f"{name}.json"), "w") as f:
            with open(os.path.join(self.data_folder, f"{name}.json"), "w") as f:
                json.dump(data, f)

        return normalized_samples

    def load_files(self, data_dir):
        samples = {}
        for file in os.listdir(data_dir):
            if file.startswith("LEACH-CE-E_") and not file.endswith("extended.json"):
                name_parts = file.split("_")[1:]
                name_parts[-1] = name_parts[-1].split(".json")[0]
                name = tuple(name_parts)
                name = tuple(float(part.replace("'", ""))
                             for part in name_parts)

                with open(os.path.join(data_dir, file), "r") as f:
                    data = json.load(f)
                samples[name] = data
        return samples

    def load_samples(self, data_dir):
        logger.info(f"Loading samples from: {data_dir}")
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

    def get_all_samples(self, samples):
        # Get the samples in the form of weights, current_membership, y_membership
        # "x_data": rnd_data,
        # "prev_x_data": prev_x_data,
        # "y_data": y_data,
        x_data_list = []
        prev_x_data_list = []
        y_data_list = []
        y_chs_list = []
        for key, sample in samples.items():
            for round in range(0, len(sample)):
                # if round == 1:
                #     continue
                x_data = sample[str(round)]['x_data']
                prev_x_data = sample[str(round)]['prev_x_data']
                y_data = sample[str(round)]['y_data']
                y_chs = sample[str(round)]['y_chs']
                x_data_list.append(x_data)
                prev_x_data_list.append(prev_x_data)
                y_data_list.append(y_data)
                y_chs_list.append(y_chs)

        return x_data_list, y_data_list, prev_x_data_list, y_chs_list

    def get_sample(self, samples, weights: tuple):
        x_data_list = []
        prev_x_data_list = []
        y_data_list = []
        for key, sample in samples.items():
            if key == weights:
                for round in range(0, len(sample)):
                    # if round == 1:
                    #     continue
                    x_data = sample[str(round)]['x_data']
                    prev_x_data = sample[str(round)]['prev_x_data']
                    y_data = sample[str(round)]['y_data']
                    x_data_list.append(x_data)
                    prev_x_data_list.append(prev_x_data)
                    y_data_list.append(y_data)

        return x_data_list, y_data_list, prev_x_data_list

    def get_model(self, load_model=False):
        model = MixedDataModel(input_dim=304,
                               hidden_dim=self.hidden_dim,
                               num_classes=101,
                               num_labels=99,
                               drop_out=self.drop_out)

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

        criterion_chs = nn.BCELoss()
        criterion_ch_assign = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate)

        return model, criterion_chs, criterion_ch_assign, optimizer

    def train(self):
        if self.model_path is None:
            raise Exception("Please provide the path to save the model")

        model, criterion_chs, criterion_ch_assign, optimizer = self.get_model()

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
                    # The cluster head target which is the last 5 elements
                    chs_target = target_data[:, -5:]
                    # The cluster assignment target which is the first 99 elements
                    chs_assign_target = target_data[:, :-5]
                    # print(
                    #     f"chs_target: {chs_target}, {chs_target.shape}")
                    # argmax to find the cluster heads
                    # chs_target_argmax = torch.argmax(
                    #     chs_target, dim=2).long()
                    # print(
                    #     f"chs_target_argmax 2: {chs_target_argmax}, {chs_target_argmax.shape}")
                    # print(
                    #     f"chs_assign_target: {chs_assign_target}, {chs_assign_target.shape}")
                    # argmax to find the membership
                    # chs_assign_target_argmax = torch.argmax(
                    #     chs_assign_target, dim=2).long()
                    # print(
                    #     f"chs_assign_target_argmax: {chs_assign_target_argmax}, {chs_assign_target_argmax.shape}")
                    # print(
                    #     f"Categorical data: {categorical_data}, {categorical_data.shape}")
                    # print(f"Target data: {target_data}, {target_data.shape}")
                    chs, chs_assign = model(
                        x=input_data, next_chs=chs_target)
                    # print(f"chs predicted: {chs}, {chs.shape}")
                    # input(
                    #     f"chs_assign predicted: {chs_assign}, {chs_assign.shape}")
                    loss_ch = criterion_chs(chs, chs_target)
                    loss_ch_assign = criterion_ch_assign(
                        chs_assign, chs_assign_target)
                    # Sum the losses
                    loss = loss_ch + loss_ch_assign
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
                        chs_target = target_data[:, -5:]
                        chs_assign_target = target_data[:, :-5]
                        chs, chs_assign = model(
                            x=input_data, next_chs=chs_target)
                        loss_ch = criterion_chs(chs, chs_target)
                        loss_ch_assign = criterion_ch_assign(
                            chs_assign, chs_assign_target)
                        # Sum the losses
                        loss = loss_ch + loss_ch_assign
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
        # Convert one hot encoded to categorical
        y = torch.argmax(y[0], dim=1)
        # _, predicted = torch.max(output.data, 1)
        # print(f"Predicted: {predicted}")
        _, predicted = torch.max(output.data[0], 1)
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

        model, criterion_chs, criterion_ch_assign, optimizer = self.get_model()

        if weights is not None:
            if batch is not None:
                self.batch_size = batch
            # Convert weights to tuple
            weights = tuple(weights)
            # Load the data
            samples = self.load_data()
            x, y, prev_x = self.get_sample(
                samples, weights=weights)
            np_x = np.array(x)
            np_y = np.array(y)
            np_prev_x = np.array(prev_x)

            # Lets one hot encode the y
            # print(f"np_y: {np_y[0]}")
            np_y = np.eye(self.num_clusters+1)[np_y.astype(int)]

            # Concatenate the weights and current_membership
            # np_x = np.concatenate(
            #     (np_x, np_prev_x), axis=1)
            self.testing_dataset = NetworkDataset(x=np_x, y=np_y)
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
                chs_target = target_data[:, -5:]
                chs_assign_target = target_data[:, :-5]
                # Print the X every 209 elements in the array
                # temp = X[0]
                # for i in range(0, len(temp), 209):
                #     print(f"X {i/209}: {temp[i:i+209]}")
                chs, chs_assign = model(
                    x=input_data, next_chs=chs_target)
                loss_ch = criterion_chs(chs, chs_target)
                loss_ch_assign = criterion_ch_assign(
                    chs_assign, chs_assign_target)
                # Sum the losses
                loss = loss_ch + loss_ch_assign
                losses.append(loss.item())
                correct, total = self.test_predicted_chs(
                    chs_target, chs, print_output)
                # acc = correct/total * 100
                # avg_accuracy.append(acc)
                # predict cluster assignment
                correct, total = self.test_predicted_ch_assign(
                    chs_assign_target, chs_assign, print_output)
        logger.info(f"Average Loss: {np.mean(losses)}")
        logger.info(f"Average Accuracy: {np.mean(avg_accuracy)}")
        logger.info(f"Accuracy Min: {np.min(avg_accuracy)}")
        logger.info(
            f"Number of samples with minimum accuracy: {np.sum(np.array(avg_accuracy) == np.min(avg_accuracy))}")

    def test_predicted_chs(self, y, output, print_output=False):
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

    def test_predicted_ch_assign(self, y, output, print_output=False):
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
            # print the values of the correct index
            logger.info(f"Correct values: {y[index]}")
            # get the index where the values are not equal
            index = np.where(y != predicted)
            logger.info(f"Incorrect index: {index}")
            logger.info(f"Correct: {correct}, Total: {total}")
            input("Press enter to continue")
        return correct, total
