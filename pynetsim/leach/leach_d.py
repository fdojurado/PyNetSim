# This is based on:
# @article{liu_leach-d_2024,
# 	title = {{LEACH}-{D}: {A} low-energy, low-delay data transmission method for industrial internet of things wireless sensors},
# 	volume = {4},
# 	shorttitle = {{LEACH}-{D}},
# 	url = {https://www.sciencedirect.com/science/article/pii/S2667345223000524},
# 	urldate = {2024-08-03},
# 	journal = {Internet of Things and Cyber-Physical Systems},
# 	author = {Liu, Desheng and Liang, Chen and Mo, Hongwei and Chen, Xiaowei and Kong, Dequan and Chen, Peng},
# 	year = {2024},
# 	note = {Publisher: Elsevier},
# 	pages = {129--137},
# 	file = {Available Version (via Google Scholar):/Users/fernando/Zotero/storage/DQFIA7RV/S2667345223000524.html:text/html;Liu et al. - 2024 - LEACH-D A low-energy, low-delay data transmission.pdf:/Users/fernando/Zotero/storage/ZZ66Q4Q6/Liu et al. - 2024 - LEACH-D A low-energy, low-delay data transmission.pdf:application/pdf},
# }


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pynetsim.common as common

from rich.progress import Progress
from pynetsim.utils import RandomNumberGenerator
from pynetsim.leach.leach import LEACH

# Lets the class inherit from the LEACH class


class LEACH_D(LEACH):

    def __init__(self, network, net_model):
        self.name = "LEACH-D"
        super().__init__(network=network, net_model=net_model, name=self.name)

    def select_cluster_heads(self, round, probability, tleft, num_cluster_heads, avg_re):
        for node in self.network:
            if self.network.should_skip_node(node):
                continue

            if node.rounds_to_become_cluster_head > 0:
                node.rounds_to_become_cluster_head -= 1

            self.network.mark_as_non_cluster_head(node)

            if self.should_select_cluster_head(round, node, probability, avg_re):
                num_cluster_heads = self.mark_as_cluster_head(
                    node, num_cluster_heads, tleft)

    def should_select_cluster_head(self, round, node, probability, avg_re):
        if round == 1:
            th = probability
        else:
            remaining_energy = node.remaining_energy
            th = probability * remaining_energy/avg_re
        return (not node.is_cluster_head and node.rounds_to_become_cluster_head == 0
                and self.rng.get_random() < th)

    def mark_as_main_cluster_head(self, node, num_main_cluster_heads):
        num_main_cluster_heads += 1
        self.network.mark_as_main_cluster_head(node, num_main_cluster_heads)
        return num_main_cluster_heads

    def should_select_main_cluster_head(self, node, probability, avg_re):
        remaining_energy = node.remaining_energy
        th = probability * remaining_energy/avg_re
        return self.rng.get_random() < th

    def select_main_cluster_heads(self, probability, num_main_cluster_heads, avg_re):
        # loop through all cluster heads
        for node in self.network:
            if self.network.should_skip_node(node):
                continue

            self.network.mark_as_non_main_cluster_head(node)

            if not node.is_cluster_head:
                continue

            if self.should_select_main_cluster_head(node, probability, avg_re):
                num_main_cluster_heads = self.mark_as_main_cluster_head(
                    node, num_main_cluster_heads)
                # print(f"Node {node.node_id} is MCH (ID: {node.mch_id})")

    def evaluate_round(self, p, round):
        round += 1
        # print(f"Evaluating round {round} of {self.name}")

        num_cluster_heads = 0
        num_main_cluster_heads = 0
        th = p / (1 - p * (round % (1 / p)))
        tleft = round % (1 / p)

        avg_re = self.network.average_remaining_energy()

        self.select_cluster_heads(
            round, th, tleft, num_cluster_heads, avg_re)
        cluster = self.network.create_clusters()
        # if not cluster:
        #     raise f"No cluster was created at round {round}"
        # At this stage we need to create the MCH (Main Cluster Heads)
        self.select_main_cluster_heads(th, num_main_cluster_heads, avg_re)
        # self.network.print_clusters()
        mch = self.network.create_mch_clusters()
        # if not mch:
        #     print(f"No MCH was created at round {round}")
        self.net_model.dissipate_energy(round=round)

        return round
