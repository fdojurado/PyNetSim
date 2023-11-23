import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import pynetsim.leach.surrogate as leach_surrogate


from torch.utils.data import Dataset
from pynetsim.utils import PyNetSimLogger
from rich.progress import Progress


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
        logger.info(f"Data shape: {self.data.shape}")
        # print info and describe the data
        logger.info(f"Data info: {self.data.info()}")
        logger.info(f"Data description: {self.data.describe()}")
        self.data_stats = pd.DataFrame(columns=['name', 'mean', 'std'])
        self.data_stats = leach_surrogate.get_mean_std_name(
            self.data, self.data_stats)
        # Get the remaining energy column
        remaining_energy_stats_dict = leach_surrogate.compute_stats(
            're', self.data['remaining_energy'])
        # Add to the data stats
        self.data_stats = pd.concat(
            [self.data_stats, pd.DataFrame(remaining_energy_stats_dict, index=[0])]).reset_index(drop=True)
        # Get the cluster_head column
        chs_dict = leach_surrogate.compute_array_stats(
            'chs', self.data['cluster_heads'])
        self.data_stats = pd.concat(
            [self.data_stats, pd.DataFrame(chs_dict, index=[0])]).reset_index(drop=True)
        # Get the energy levels column
        energy_levels_dict = leach_surrogate.compute_array_stats(
            'el', self.data['energy_levels'])
        self.data_stats = pd.concat(
            [self.data_stats, pd.DataFrame(energy_levels_dict, index=[0])]).reset_index(drop=True)
        energy_dissipated_ch_to_sink_dict = leach_surrogate.compute_array_stats(
            'ed_ch_to_sink', self.data['energy_dissipated_ch_to_sink'])
        self.data_stats = pd.concat(
            [self.data_stats, pd.DataFrame(energy_dissipated_ch_to_sink_dict, index=[0])]).reset_index(drop=True)
        energy_dissipated_non_ch_to_ch_dict = leach_surrogate.compute_array_stats(
            'ed_non_ch_to_ch', self.data['energy_dissipated_non_ch_to_ch'])
        self.data_stats = pd.concat(
            [self.data_stats, pd.DataFrame(energy_dissipated_non_ch_to_ch_dict, index=[0])]).reset_index(drop=True)
        logger.info(f"Data stats: {self.data_stats}")

        self.model = self.load_model()

        self.estimate_tx_energy = leach_surrogate.get_estimate_tx_energy(
            network=self.network, eelect=self.eelect, eamp=self.eamp,
            efs=self.efs, eda=self.eda, packet_size=self.packet_size, d0=self.d0)

        self.alpha, self.beta, self.gamma = leach_surrogate.get_standardized_weights(
            alpha_val=self.alpha, beta_val=self.beta, gamma_val=self.gamma,
            data_stats=self.data_stats)
        print(f"Alpha: {self.alpha}, Beta: {self.beta}, Gamma: {self.gamma}")

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
            mean=self.data_stats.loc[self.data_stats['name']
                                     == 'ed_ch_to_sink']['mean'].values[0],
            std=self.data_stats.loc[self.data_stats['name'] == 'ed_ch_to_sink']['std'].values[0])

    def get_standardized_energy_dissipated_non_ch_to_ch(self, cluster_heads: list):
        # Get the energy dissipated by cluster heads to sink
        tx_energy_to_ch = {}
        for node in self.network:
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
        assert len(
            tx_energy_to_ch_list) == 495, f"len(tx_energy_to_ch_list): {len(tx_energy_to_ch_list)}"
        return leach_surrogate.standardize_inputs(
            x=tx_energy_to_ch_list,
            mean=self.data_stats.loc[self.data_stats['name']
                                     == 'ed_non_ch_to_ch']['mean'].values[0],
            std=self.data_stats.loc[self.data_stats['name'] == 'ed_non_ch_to_ch']['std'].values[0])

    def get_standardized_cluster_heads(self, cluster_heads: list):
        return leach_surrogate.standardize_inputs(
            x=cluster_heads,
            mean=self.data_stats.loc[self.data_stats['name']
                                     == 'chs']['mean'].values[0],
            std=self.data_stats.loc[self.data_stats['name'] == 'chs']['std'].values[0])

    def predict(self, network: object, cluster_heads: list):
        # Get standardized inputs
        remaining_energy = leach_surrogate.get_standardized_remaining_energy(
            network.remaining_energy(), self.data_stats)
        logger.info(f"Standardized remaining energy: {remaining_energy}")
        # Get the energy levels
        energy_levels = leach_surrogate.get_standardized_energy_levels(
            network, self.data_stats)
        logger.info(f"Standardized energy levels: {energy_levels}")
        # Get the energy dissipated by cluster heads to sink
        edc = self.get_standardized_energy_dissipated_ch_to_sink(
            cluster_heads=cluster_heads)
        logger.info(
            f"Standardized energy dissipated by cluster heads to sink: {edc}")
        # Get the energy dissipated by non cluster heads to cluster heads
        ednc = self.get_standardized_energy_dissipated_non_ch_to_ch(
            cluster_heads=cluster_heads)
        logger.info(
            f"Standardized energy dissipated by non cluster heads to cluster heads: {ednc}")
        # Get the cluster heads
        cluster_heads.insert(0, 0)
        cluster_heads = self.get_standardized_cluster_heads(
            cluster_heads=cluster_heads)
        logger.info(f"Standardized cluster heads: {cluster_heads}")
        # Now lets put everything together
        inputs = [self.alpha, self.beta, self.gamma, remaining_energy]
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
            print(f"Outputs shape: {outputs.shape}")
            outputs = torch.argmax(outputs, dim=2)
            print(f"Output: {outputs}")
        return outputs.cpu().numpy()[0]
