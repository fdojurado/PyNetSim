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
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import pynetsim.leach.surrogate as leach_surrogate
import logging


from torch.utils.data import Dataset

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

        self.sigmoid = nn.Sigmoid()

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
        out = self.sigmoid(out)
        # print(f"out shape3: {out.shape}")

        return out


class ClusterHeadModel:

    def __init__(self, config, network=object):
        self.name = "Cluster Head Model"
        self.network = network
        self.cluster_head_percentage = config.network.protocol.cluster_head_percentage
        self.model_path = config.surrogate.cluster_head_model
        self.data_folder = config.surrogate.cluster_head_data
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

        remaining_energy_dict = leach_surrogate.compute_stats(
            data=self.data['remaining_energy'])

        self.re_mean = remaining_energy_dict['mean']
        self.re_std = remaining_energy_dict['std']

        energy_levels_dict = leach_surrogate.compute_array_stats(
            data=self.data['energy_levels'])

        self.el_mean = energy_levels_dict['mean']
        self.el_std = energy_levels_dict['std']

        energy_dissipated_ch_to_sink_dict = leach_surrogate.compute_array_stats(
            self.data['energy_dissipated_ch_to_sink'])

        self.ed_ch_to_sink_mean = energy_dissipated_ch_to_sink_dict['mean']
        self.ed_ch_to_sink_std = energy_dissipated_ch_to_sink_dict['std']

        energy_dissipated_ch_rx_from_non_ch_dict = leach_surrogate.compute_array_stats(
            self.data['energy_dissipated_ch_rx_from_non_ch'])

        self.ed_ch_rx_from_non_ch_mean = energy_dissipated_ch_rx_from_non_ch_dict['mean']
        self.ed_ch_rx_from_non_ch_std = energy_dissipated_ch_rx_from_non_ch_dict['std']

        self.model = self.load_model()

        self.estimate_tx_energy = leach_surrogate.get_estimate_tx_energy(
            network=self.network, eelect=self.eelect, eamp=self.eamp,
            efs=self.efs, eda=self.eda, packet_size=self.packet_size, d0=self.d0)

    def load_model(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        model = ForecastCCH(
            input_size=401, h1=2000, h2=2000, output_size=101).to(self.device)
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
            logger.info(f"Loaded model from {self.model_path}")
        else:
            raise Exception(f"Model not found at {self.model_path}")
        return model

    def get_standardized_potential_cluster_heads(self, network: object, avg_re: float):
        potential_cluster_heads = []
        for node in network:
            if network.should_skip_node(node):
                continue
            if node.remaining_energy >= avg_re:
                potential_cluster_heads.append(node.node_id)
        np_potential_cluster_heads = np.zeros(99)
        for ch in potential_cluster_heads:
            np_potential_cluster_heads[ch-2] = 1
        np_potential_cluster_heads = list(np_potential_cluster_heads)
        return {'pchs': potential_cluster_heads,
                'pchs_one_hot': np_potential_cluster_heads}

    def get_standardized_energy_dissipated_ch_to_sink(self, pchs: list):
        # Get the energy dissipated by cluster heads to sink
        np_ch_to_sink = np.zeros(99)
        for ch in pchs:
            np_ch_to_sink[ch-2] = self.estimate_tx_energy[ch][1]
        ch_to_sink = list(np_ch_to_sink)
        # standardize the data
        return leach_surrogate.standardize_inputs(
            x=ch_to_sink,
            mean=self.ed_ch_to_sink_mean,
            std=self.ed_ch_to_sink_std)

    def get_standardized_energy_dissipated_ch_rx_from_non_ch(self, potential_cluster_heads,
                                                             alive_nodes: int):
        np_ch_from_non_ch = np.zeros(99)
        estimated_num_chs = int(
            alive_nodes*self.cluster_head_percentage) + 1
        if estimated_num_chs == 0:
            estimated_num_chs = 1
        else:
            num_non_ch_per_ch = alive_nodes/estimated_num_chs
        for ch in potential_cluster_heads:
            ch_rx_energy = self.eelect * self.packet_size * \
                num_non_ch_per_ch
            np_ch_from_non_ch[ch-2] = ch_rx_energy
        ch_from_non_ch = list(np_ch_from_non_ch)
        return leach_surrogate.standardize_inputs(
            x=ch_from_non_ch,
            mean=self.ed_ch_rx_from_non_ch_mean,
            std=self.ed_ch_rx_from_non_ch_std)

    def get_standardized_num_cluster_heads(self, alive_nodes: int):
        num_chs = (int(alive_nodes*self.cluster_head_percentage)+1)/5
        return num_chs

    def predict(self, network: object, round: int, std_re: float, std_el: list,
                avg_re: float, alive_nodes: int):
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
        # Get potential cluster heads
        potential_cluster_heads = self.get_standardized_potential_cluster_heads(
            network, avg_re=avg_re)
        # logger.debug(
        #     f"Potential cluster heads: {potential_cluster_heads['pchs']} one hot: {potential_cluster_heads['pchs_one_hot']}")
        # Get the energy levels
        energy_levels = std_el
        # logger.debug(f"Standardized energy levels: {energy_levels}")
        # Get the energy dissipated by cluster heads to sink
        edc = self.get_standardized_energy_dissipated_ch_to_sink(
            pchs=potential_cluster_heads['pchs'])
        # logger.debug(
        #     f"Standardized energy dissipated by cluster heads to sink: {edc}")
        # Get the energy consumed by non-cluster heads when transmitting to cluster heads
        # ednch = leach_surrogate.get_standardized_energy_dissipated_non_ch_to_ch(
        #     potential_cluster_heads=potential_cluster_heads['pchs'],
        #     network=network,
        #     estimate_tx_energy=self.estimate_tx_energy,
        #     data_stats=self.data_stats)
        # logger.info(
        #     f"Standardized energy dissipated by non-cluster heads when transmitting to cluster heads: {ednch}")
        # Get the energy dissipated by chs when receiving from non-chs
        energy_dissipated_ch_rx_from_non_ch = self.get_standardized_energy_dissipated_ch_rx_from_non_ch(
            potential_cluster_heads['pchs'], alive_nodes=alive_nodes)
        # logger.debug(
        #     f"Standardized energy dissipated by cluster heads when receiving from non-cluster heads: {energy_dissipated_ch_rx_from_non_ch}")
        # Get the number of cluster heads
        num_cluster_heads = self.get_standardized_num_cluster_heads(
            alive_nodes=alive_nodes)
        # logger.debug(
        #     f"Standardized number of cluster heads: {num_cluster_heads}")
        # Now lets put everything together
        inputs = self.weights + [remaining_energy]
        inputs.extend(potential_cluster_heads['pchs_one_hot'])
        inputs.extend(energy_levels)
        inputs.extend(edc)
        # inputs.extend(ednch)
        inputs.extend(energy_dissipated_ch_rx_from_non_ch)
        inputs.append(num_cluster_heads)
        # Lets predict the cluster heads
        inputs = np.array(inputs)
        inputs = inputs.reshape(1, -1)
        inputs = torch.from_numpy(inputs.astype(np.float32))
        inputs = inputs.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            outputs = outputs.cpu().numpy()
            outputs = outputs.reshape(-1)
            # logger.info(f"Predicted cluster heads: {outputs}")
            predicted = outputs > 0.5
            predicted = np.where(predicted == True)[0]
            # print(f"Predicted cluster heads 1: {predicted}")
            # get the number of cluster heads excluding the ch 0
            count_heads_count = 0
            for ch in predicted:
                if ch != 0:
                    count_heads_count += 1
            # logger.info(
            #     f"Expected number of cluster heads: {num_cluster_heads*5}")
            #  if len(predicted) < num_cluster_heads then we need to add more cluster heads
            #  the cluster head with the highest probability will be selected that is not already a cluster head
            if count_heads_count < num_cluster_heads*5:
                # if there is cluster id 0 then remove it
                if 0 in predicted:
                    predicted = np.delete(predicted, 0)
                # How many more cluster heads do we need?
                num_more_chs = num_cluster_heads*5 - count_heads_count
                num_more_chs = int(num_more_chs)
                indices = outputs.argsort()[::-1]
                count_more_chs = 0
                while count_more_chs < num_more_chs:
                    for index in indices:
                        if index == 0:
                            continue
                        if index not in predicted:
                            predicted = np.insert(predicted, 0, index)
                            count_more_chs += 1
                            break
                # sort the predicted cluster heads
                predicted.sort()
                # print(f"Predicted cluster heads 2: {predicted}")
                # input("Press enter to continue...")
            if len(predicted) > 5:
                predicted = outputs.argsort()[-5:][::-1]
                # logger.info(f"Predicted cluster heads 2: {predicted}")
            if len(predicted) < 5:
                predicted = np.insert(predicted, 0, np.zeros(5-len(predicted)))
            # convert to a list and sort
            predicted = list(predicted)
            predicted.sort()
            # logger.info(f"Predicted cluster heads 3: {predicted}")
        # See if the predicted cluster heads are the same as the actual cluster heads
        # Get the actual cluster heads
        # actual = data_at_round['cluster_heads']
        # actual = eval(actual)
        # Get the actual energy levels
        # actual_energy_levels = data_at_round['energy_levels']
        # actual_energy_levels = eval(actual_energy_levels)
        # check if the predicted cluster heads are the same as the actual cluster heads
        # if predicted == actual:
            # logger.info(
            #     f"Predicted cluster heads are the same as the actual cluster heads.")
            # # print cluster heads, round, alive nodes and current energy of all nodes
            # logger.info(f"Predicted cluster heads: {predicted}")
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
        #         f"Predicted cluster heads are not the same as the actual cluster heads.")
        #     logger.info(f"Predicted cluster heads: {predicted}")
        #     logger.info(f"Actual cluster heads: {actual}")
        #     logger.info(f"Round: {round}")
        #     logger.info(f"Alive nodes: {network.alive_nodes()}")
        #     # check if the energy levels are the same
        #     if np.all(np.array(actual_energy_levels) == np.array(energy_levels)):
        #         logger.info(
        #             f"Energy levels are the same as the actual energy levels.")
        #     else:
        #         # print the energy levels that are different
        #         for i, energy_level in enumerate(actual_energy_levels):
        #             if energy_level != energy_levels[i]:
        #                 logger.info(
        #                     f"Node {i+2} energy level is different. Predicted: {energy_levels[i]}, Actual: {energy_level}")
        #     input("Press Enter to continue...")
        return predicted
