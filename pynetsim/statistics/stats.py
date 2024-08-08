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


import json
import os
import errno


class Statistics(object):

    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.save_path = self.config.save_path
        try:
            os.makedirs(self.save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self._round_stats = {}
        self.__name = self.config.network.protocol.name + '_' + \
            self.config.network.model

    def add_round_stats(self, round, remaining_energy, dead_nodes, alive_nodes,
                        num_cluster_heads, pdr, plr,
                        control_packets_energy, control_pkt_bits, pkts_sent_to_bs,
                        energy_dissipated, pkts_recv_by_bs, membership,
                        energy_levels, cluster_heads, dst_to_cluster_head):

        self._round_stats[round] = {
            'remaining_energy': remaining_energy,
            'dead_nodes': dead_nodes,
            'alive_nodes': alive_nodes,
            'num_cluster_heads': num_cluster_heads,
            'pdr': pdr,
            'plr': plr,
            'control_packets_energy': control_packets_energy,
            'control_pkt_bits': control_pkt_bits,
            'pkts_sent_to_bs': pkts_sent_to_bs,
            'energy_dissipated': energy_dissipated,
            'pkts_recv_by_bs': pkts_recv_by_bs,
            'membership': membership,
            'energy_levels': energy_levels,
            'cluster_heads': cluster_heads,
            'dst_to_cluster_head': dst_to_cluster_head
        }

    # This function is called when a round is finished, so we generate the
    # statistics for the round
    def generate_round_stats(self, round):
        remaining_energy = self.network.remaining_energy()
        dead_nodes = self.network.dead_nodes()
        alive_nodes = self.network.alive_nodes()
        num_cluster_heads = self.network.num_cluster_heads()
        pdr = self.network.average_pdr()
        plr = self.network.average_plr()
        control_packets_energy = self.network.control_packets_energy()
        control_pkt_bits = self.network.control_pkt_bits()
        pkts_sent_to_bs = self.network.pkts_sent_to_bs()
        energy_dissipated = self.network.energy_dissipated()
        pkts_recv_by_bs = self.network.pkts_recv_by_bs()
        # Put membership of each node in the cluster
        membership = {}
        for node in self.network:
            membership[node.node_id] = node.cluster_id
        # Put the energy level of each node excluding the sink
        energy_levels = {}
        dst_to_cluster_head = {}
        for node in self.network:
            if node.node_id != 1:
                energy_levels[node.node_id] = node.remaining_energy
                dst_to_cluster_head[node.node_id] = node.dst_to_cluster_head
        # Put the cluster heads at this round
        cluster_heads = self.network.get_cluster_head_ids()

        self.add_round_stats(round, remaining_energy, dead_nodes, alive_nodes,
                             num_cluster_heads, pdr,
                             plr, control_packets_energy,
                             control_pkt_bits, pkts_sent_to_bs,
                             energy_dissipated, pkts_recv_by_bs,
                             membership, energy_levels, cluster_heads,
                             dst_to_cluster_head)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    def get_round_stats(self, round):
        return self._round_stats[round]

    def get_all_round_stats(self):
        return self._round_stats

    def get_rounds(self):
        return self._round_stats.keys()

    def get_remaining_energy(self, round):
        return self._round_stats[round]['remaining_energy']

    def get_dead_nodes(self, round):
        return self._round_stats[round]['dead_nodes']

    def get_alive_nodes(self, round):
        return self._round_stats[round]['alive_nodes']

    def export_json(self):
        # If the results directory does not exist, create it
        try:
            os.makedirs('results')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        with open(self.save_path + self.name + '.json', 'w') as outfile:
            json.dump(self._round_stats, outfile)
