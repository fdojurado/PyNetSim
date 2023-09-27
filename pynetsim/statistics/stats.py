import json
import os
import errno


class Statistics(object):

    def __init__(self, network, config):
        self.network = network
        self.config = config
        self._round_stats = {}
        self.__name = self.config.network.protocol.name + '_' + \
            self.config.network.model

    def add_round_stats(self, round, remaining_energy, dead_nodes, alive_nodes,
                        num_cluster_heads, pdr, plr,
                        control_packets_energy, control_pkt_bits, pkts_sent_to_bs,
                        energy_dissipated, pkts_recv_by_bs, membership,
                        energy_levels, cluster_heads):

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
            'cluster_heads': cluster_heads
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
        for node in self.network:
            if node.node_id != 1:
                energy_levels[node.node_id] = node.remaining_energy
        # Put the cluster heads at this round
        cluster_heads = self.network.get_cluster_head_ids()

        self.add_round_stats(round, remaining_energy, dead_nodes, alive_nodes,
                             num_cluster_heads, pdr,
                             plr, control_packets_energy,
                             control_pkt_bits, pkts_sent_to_bs,
                             energy_dissipated, pkts_recv_by_bs,
                             membership, energy_levels, cluster_heads)

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
        with open('results/' + self.name + '.json', 'w') as outfile:
            json.dump(self._round_stats, outfile)
