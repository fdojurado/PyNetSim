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

# This is based on:
# @article{bharany_energy-efficient_2021,
# 	title = {Energy-efficient clustering scheme for flying ad-hoc networks using an optimized {LEACH} protocol},
# 	volume = {14},
# 	url = {https://www.mdpi.com/1996-1073/14/19/6016},
# 	number = {19},
# 	urldate = {2024-08-03},
# 	journal = {Energies},
# 	author = {Bharany, Salil and Sharma, Sandeep and Badotra, Sumit and Khalaf, Osamah Ibrahim and Alotaibi, Youseef and Alghamdi, Saleh and Alassery, Fawaz},
# 	year = {2021},
# 	note = {Publisher: MDPI},
# 	pages = {6016},
# 	file = {Available Version (via Google Scholar):/Users/fernando/Zotero/storage/CM6UW2IA/Bharany et al. - 2021 - Energy-efficient clustering scheme for flying ad-h.pdf:application/pdf},
# }


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pynetsim.common as common

from rich.progress import Progress
from pynetsim.utils import RandomNumberGenerator
from pynetsim.leach.leach import LEACH

# Lets the class inherit from the LEACH class


class LEACH_EE(LEACH):

    def __init__(self, network, net_model):
        self.name = "LEACH-EE"
        super().__init__(network=network, net_model=net_model, name=self.name)

    def select_cluster_heads(self, probability, tleft, num_cluster_heads, avg_drain_rate, avg_re):
        for node in self.network:
            if self.network.should_skip_node(node):
                continue

            if node.rounds_to_become_cluster_head > 0:
                node.rounds_to_become_cluster_head -= 1

            self.network.mark_as_non_cluster_head(node)

            if self.should_select_cluster_head(node, probability, avg_drain_rate, avg_re):
                num_cluster_heads = self.mark_as_cluster_head(
                    node, num_cluster_heads, tleft)

    def should_select_cluster_head(self, node, probability, avg_drain_rate, avg_re):
        remaining_energy = node.remaining_energy
        drain_rate = node.drain_rate
        th = probability * (remaining_energy-drain_rate) / \
            (avg_re-avg_drain_rate)
        return (not node.is_cluster_head and node.rounds_to_become_cluster_head == 0
                and self.rng.get_random() < th)

    def evaluate_round(self, p, round):
        print(f"Evaluating round {round} of {self.name}")
        round += 1

        num_cluster_heads = 0
        th = p / (1 - p * (round % (1 / p)))
        tleft = round % (1 / p)

        avg_drain_rate = self.network.average_drain_rate()
        avg_re = self.network.average_remaining_energy()

        self.select_cluster_heads(
            th, tleft, num_cluster_heads, avg_drain_rate, avg_re)
        self.network.create_clusters()
        self.net_model.dissipate_energy(round=round)

        return round
