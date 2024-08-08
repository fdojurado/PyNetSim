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

import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import pynetsim.leach.surrogate as leach_surrogate
import logging

from torch.utils.data import Dataset
from rich.progress import Progress


# -------------------- Create logger --------------------
logger = logging.getLogger("Main")



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
    global non_cyclical_features_size, cyclical_features_size

    def __init__(self, input_size=10, h1=100, h2=100, output_size=101):
        super(ForecastCCH, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, h1)
        self.batch_norm2 = nn.BatchNorm1d(h1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(h1, h2)
        self.batch_norm3 = nn.BatchNorm1d(h2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(h2, output_size)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        out = self.batch_norm1(x)
        out = self.fc1(x)
        out = self.batch_norm2(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        # print(f"out shape1: {out.shape}")

        out = self.fc2(out)
        out = self.batch_norm3(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        # print(f"out shape2: {out.shape}")

        out = self.fc3(out)
        # out = self.sigmoid(out)
        # print(f"out shape3: {out.shape}")
        # reshape output to [batch_size, 99, 101]
        out = out.view(-1, 99, 6)
        return out


class ClusterAssignmentModel:

    def __init__(self, config, network: object):
        self.name = "Cluster Assignment Model"
        self.network = network
        self.model_path = config.surrogate.cluster_assignment_model
        self.data_folder = config.surrogate.cluster_assignment_data
        self.alpha = config.surrogate.alpha
        self.beta = config.surrogate.beta
        self.gamma = config.surrogate.gamma
        self.eelect = config.network.protocol.eelect
        self.eamp = config.network.protocol.eamp
        self.efs = config.network.protocol.efs
        self.eda = config.network.protocol.eda
        self.packet_size = config.network.protocol.packet_size
        self.d0 = (self.efs/self.eamp)**0.5
        # assert if model and data folder are provided
        if self.model_path is None:
            raise Exception(
                f"{self.name}: Please provide the path to the model")
        if self.data_folder is None:
            raise Exception(
                f"{self.name}: Please provide the path to the data")
        self.init()

    def init(self):
        self.data = pd.read_csv(self.data_folder)
        # logger.info(f"Data shape: {self.data.shape}")
        # print info and describe the data
        # logger.info(f"Data info: {self.data.info()}")
        # logger.info(f"Data description: {self.data.describe()}")
        weights_dict = leach_surrogate.get_mean_std_name(
            data=self.data)
        alpha_standardized = leach_surrogate.standardize_inputs(
            x=self.alpha,
            mean=weights_dict['alpha']['mean'],
            std=weights_dict['alpha']['std'])
        beta_standardized = leach_surrogate.standardize_inputs(
            x=self.beta,
            mean=weights_dict['beta']['mean'],
            std=weights_dict['beta']['std'])
        gamma_standardized = leach_surrogate.standardize_inputs(
            x=self.gamma,
            mean=weights_dict['gamma']['mean'],
            std=weights_dict['gamma']['std'])
        self.weights = [alpha_standardized, beta_standardized,
                        gamma_standardized]
        # Get the cluster_head column
        chs_dict = leach_surrogate.compute_array_stats(
            self.data['cluster_heads'])

        self.chs_mean = chs_dict['mean']
        self.chs_std = chs_dict['std']

        energy_dissipated_ch_to_sink_dict = leach_surrogate.compute_array_stats(
            self.data['energy_dissipated_ch_to_sink'])

        self.ed_ch_to_sink_mean = energy_dissipated_ch_to_sink_dict['mean']
        self.ed_ch_to_sink_std = energy_dissipated_ch_to_sink_dict['std']

        energy_dissipated_non_ch_to_ch_dict = leach_surrogate.compute_array_stats(
            self.data['energy_dissipated_non_ch_to_ch'])

        self.ed_non_ch_to_ch_mean = energy_dissipated_non_ch_to_ch_dict['mean']
        self.ed_non_ch_to_ch_std = energy_dissipated_non_ch_to_ch_dict['std']

        self.model = self.load_model()

        self.estimate_tx_energy = leach_surrogate.get_estimate_tx_energy(
            network=self.network, eelect=self.eelect, eamp=self.eamp,
            efs=self.efs, eda=self.eda, packet_size=self.packet_size, d0=self.d0)

    def load_model(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        model = ForecastCCH(
            input_size=703, h1=2000, h2=2000, output_size=6*99).to(self.device)
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
            logger.info(f"Loaded model from {self.model_path}")
        else:
            raise Exception(f"Model not found at {self.model_path}")
        return model

    def get_standardized_energy_dissipated_ch_to_sink(self, cluster_heads: list):
        np_ch_to_sink = np.zeros(99)
        for ch in cluster_heads:
            if ch <= 1:
                continue
            np_ch_to_sink[ch-2] = self.estimate_tx_energy[ch][1]
        ch_to_sink = list(np_ch_to_sink)
        # standardize the data
        return leach_surrogate.standardize_inputs(
            x=ch_to_sink,
            mean=self.ed_ch_to_sink_mean,
            std=self.ed_ch_to_sink_std)

    def get_standardized_energy_dissipated_non_ch_to_ch(self, network: object, cluster_heads: list):
        # Get the energy dissipated by cluster heads to sink
        tx_energy_to_ch = {}
        for node in network:
            if node.node_id <= 1:
                continue
            if node.node_id in cluster_heads:
                tx_energy_to_ch[node.node_id] = [0, 0, 0, 0, 0]
                continue
            # Check if the node has energy
            if node.remaining_energy <= 0:
                tx_energy_to_ch[node.node_id] = [0, 0, 0, 0, 0]
                continue
            cluster_head_energy = []
            for ch in cluster_heads:
                if ch == 0:
                    cluster_head_energy.append(0)
                    continue
                cluster_head_energy.append(
                    self.estimate_tx_energy[node.node_id][ch])
            tx_energy_to_ch[node.node_id] = cluster_head_energy
        tx_energy_to_ch_list = [value for _, value in tx_energy_to_ch.items(
        )]
        tx_energy_to_ch_list = [
            item for sublist in tx_energy_to_ch_list for item in sublist]
        return leach_surrogate.standardize_inputs(
            x=tx_energy_to_ch_list,
            mean=self.ed_non_ch_to_ch_mean,
            std=self.ed_non_ch_to_ch_std)

    def get_standardized_cluster_heads(self, cluster_heads: list):
        return leach_surrogate.standardize_inputs(
            x=cluster_heads,
            mean=self.chs_mean,
            std=self.chs_std)

    def predict(self, network: object, cluster_heads: list, std_re: float, std_el: list, round: int):
        # print the cluster heads at that round in the data
        # data_at_round = self.data[(self.data['name'].str.contains(
        #     str(self.alpha))) & (self.data['name'].str.contains(str(self.beta))) & (self.data['name'].str.contains(str(self.gamma)))]
        #  reset index
        # data_at_round = data_at_round.reset_index(drop=True)
        # print(f"Data at round {round}: {data_at_round}")
        # Get the data at round
        # data_at_round = data_at_round.loc[round-1]
        # print(f"Data at round {round}: {data_at_round}")
        # Get standardized inputs
        remaining_energy = std_re
        # logger.debug(f"Standardized remaining energy: {remaining_energy}")
        # Get the energy levels
        energy_levels = std_el
        # logger.debug(f"Standardized energy levels: {energy_levels}")
        # Get the energy dissipated by cluster heads to sink
        edc = self.get_standardized_energy_dissipated_ch_to_sink(
            cluster_heads=cluster_heads)
        # logger.debug(
        #     f"Standardized energy dissipated by cluster heads to sink: {edc}")
        # Get the energy dissipated by non cluster heads to cluster heads
        ednc = self.get_standardized_energy_dissipated_non_ch_to_ch(
            network=network,
            cluster_heads=cluster_heads)
        # logger.debug(
        #     f"Standardized energy dissipated by non cluster heads to cluster heads: {ednc}")
        # Get the cluster heads
        cluster_heads.insert(0, 0)
        cluster_heads_original = cluster_heads.copy()
        cluster_heads = self.get_standardized_cluster_heads(
            cluster_heads=cluster_heads)
        # logger.debug(f"Standardized cluster heads: {cluster_heads}")
        # Now lets put everything together
        inputs = self.weights + [remaining_energy]
        inputs.extend(energy_levels)
        inputs.extend(edc)
        inputs.extend(ednc)
        inputs.extend(cluster_heads)
        inputs = np.array(inputs)
        inputs = inputs.reshape(1, -1)
        inputs = torch.from_numpy(inputs.astype(np.float32))
        inputs = inputs.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            # print(f"Outputs shape: {outputs.shape}")
            outputs = torch.argmax(outputs, dim=2)
        # Now match each element in cluster assignments whose value is the index of the cluster head
        # convert to numpy
        outputs = outputs.cpu().numpy()[0]
        # print(f"Output: {outputs}")
        # print cluster heads
        # print(f"Cluster heads original: {cluster_heads_original}")
        for i, cluster_head in enumerate(outputs):
            outputs[i] = cluster_heads_original[cluster_head]
        # print(f"Predicted cluster heads: {outputs}")
        # actual = data_at_round['membership']
        # actual = eval(actual)
        # actual_energy_levels = data_at_round['energy_levels']
        # actual_energy_levels = eval(actual_energy_levels)
        # compare the predicted membership with the actual membership using all or any
        # if np.all(outputs == actual):
            # logger.info(
            #     f"Predicted membership is the same as the actual membership.")
            # # print cluster heads, round, alive nodes and current energy of all nodes
            # logger.info(f"Predicted membership: {outputs}")
            # logger.info(f"Round: {round}")
            # logger.info(f"Alive nodes: {network.alive_nodes()}")
            # for node in network:
            #     if network.should_skip_node(node):
            #         continue
            #     logger.info(
            #         f"Node {node.node_id} current energy: {node.remaining_energy}")
            # input("Press Enter to continue...")
        #     pass
        # else:
        #     logger.info(
        #         f"Predicted membership is different from the actual membership.")
        #     logger.info(f"Predicted membership: {outputs}")
        #     logger.info(f"Actual cluster heads: {actual}")
        #     logger.info(f"Round: {round}")
        #     logger.info(f"Alive nodes: {network.alive_nodes()}")
        #     # print which value is different
        #     for i, cluster_head in enumerate(outputs):
        #         if cluster_head != actual[i]:
        #             logger.info(
        #                 f"Node {i+2} is assigned to {cluster_head} but should be assigned to {actual[i]}.")
        #     # check if the energy levels are the same
        #     if np.all(np.array(actual_energy_levels) == np.array(energy_levels)):
        #         logger.info(
        #             f"Energy levels are the same as the actual energy levels.")
        #     else:
        #         # print the energy levels that are different
        #         for i, energy_level in enumerate(actual_energy_levels):
        #             # standardize the energy level
        #             energy_level = leach_surrogate.standardize_inputs(
        #                 x=energy_level,
        #                 mean=0.24285118165286126,
        #                 std=0.01233563157394424)
        #             if energy_level != energy_levels[i]:
        #                 logger.info(
        #                     f"Node {i+2} energy level is different. Predicted: {energy_levels[i]}, Actual: {energy_level}")
        #     input("Press Enter to continue...")
        return outputs
