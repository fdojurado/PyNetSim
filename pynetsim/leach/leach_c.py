import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rich.progress import Progress


class LEACH_C:

    def __init__(self, network, net_model: object):
        self.name = "LEACH-C"
        self.net_model = net_model
        self.config = network.config
        self.network = network

    def potential_cluster_heads(self, network_avg_energy):
        potential_cluster_heads = []
        # Elect cluster heads
        for node in self.network:
            if self.network.should_skip_node(node):
                continue

            self.network.mark_as_non_cluster_head(node)

            if node.energy >= network_avg_energy:
                potential_cluster_heads.append(node.node_id)

        # print(f"Cluster heads candidates: {potential_cluster_heads}")

        return potential_cluster_heads

    def node_distance_to_cluster_candidate(self, node,
                                           cluster_head_assignments):
        # if cluster_head_assignments is empty, then return the distance to the sink
        if len(cluster_head_assignments) == 0:
            return node.dst_to_sink
        cluster_heads = [
            self.network.get_node(cluster_head_id)
            for cluster_head_id in cluster_head_assignments
        ]
        distances = {
            cluster_head.node_id: (
                (node.x - cluster_head.x)**2 + (node.y - cluster_head.y)**2)**0.5
            for cluster_head in cluster_heads
        }
        cluster_head_id = min(distances, key=distances.get)
        return distances[cluster_head_id]

    def objective_function(self, cluster_heads):
        # print(f"Calculating objective function for {cluster_heads}")
        total_sum_square_distance = 0

        for node in self.network:
            if node.node_id in cluster_heads:
                # Skip nodes that are not cluster heads
                continue
            # Assign cluster head
            min_distance = self.node_distance_to_cluster_candidate(
                node, cluster_heads)
            # Compute the total sum square distance
            total_sum_square_distance += min_distance**2

        return total_sum_square_distance

    def simulated_annealing(self, cluster_heads):
        # Initial solution
        # print(f"Cluster heads: {cluster_heads}, length: {len(cluster_heads)}")
        # Select randomly from the cluster heads without repeating
        best = np.random.choice(
            cluster_heads, size=int(len(cluster_heads)/2),
            replace=False)

        # print(
        #     f"Initial solution: {initial_solution}, length: {len(initial_solution)}")

        # Evaluate the initial solution
        best_eval = self.objective_function(best)

        # current working solution
        curr, curr_eval = best, best_eval

        initial_temp = 10

        # Run the algorithm for 1000 iterations
        for i in range(100):
            # The cluster heads contains the cluster heads candidates
            # Add a cluster head to the current solution from the cluster heads
            # or remove a cluster head from the current solution

            # Select a random cluster head
            cluster_head = np.random.choice(cluster_heads)

            # If the cluster head is in the current solution, then remove it
            if cluster_head in curr:
                candidate = np.delete(curr, np.where(curr == cluster_head))
            else:
                # If the cluster head is not in the current solution, then add it
                candidate = np.append(curr, cluster_head)

            # Evaluate the current solution
            candidate_eval = self.objective_function(candidate)

            # If the current solution is better than the initial solution, then
            # accept the current solution
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
            # Report progress
            # print(
            #     f"Best solution: {initial_solution}, length: {len(initial_solution)}")

            diff = candidate_eval - curr_eval

            t = initial_temp / (i + 1)
            metropolis = np.exp(-diff / t)
            if diff < 0 or np.random.rand() < metropolis:
                curr, curr_eval = candidate, candidate_eval

        # print(
        #     f"Best solution: {initial_solution}, length: {len(initial_solution)}")

        # Assign the cluster heads
        for node in self.network:
            if node.node_id in best:
                self.num_cluster_heads += 1
                self.network.mark_as_cluster_head(
                    node, self.num_cluster_heads)

    def choose_cluster_heads(self, chs):
        # Use simulated annealing to choose the cluster heads
        self.simulated_annealing(chs)

    def run(self):
        print("Running LEACH_C...")
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = False

        for node in self.network:
            node.is_cluster_head = False

        network_energy = {}
        num_dead_nodes = {}
        num_alive_nodes = {}
        num_cluster_heads = {}
        pkt_delivery_ratio = {}
        pkt_loss_ratio = {}
        control_packets_energy = {}
        control_packet_bits = {}
        pkts_sent_to_bs = {}
        energy_dissipated = {}

        # Set all dst_to_sink for all nodes
        for node in self.network:
            node.dst_to_sink = self.network.distance_to_sink(node)

        if not plot_clusters_flag:
            self.run_without_plotting(
                num_rounds, network_energy, num_dead_nodes, num_alive_nodes,
                num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                control_packets_energy, control_packet_bits,
                pkts_sent_to_bs,
                energy_dissipated)
        else:
            self.run_with_plotting(
                num_rounds, network_energy, num_dead_nodes, num_alive_nodes,
                num_cluster_heads, pkt_delivery_ratio, pkt_loss_ratio,
                control_packets_energy, control_packet_bits,
                pkts_sent_to_bs,
                energy_dissipated)

        common.plot_metrics(network_energy, "Network Energy", "J",
                            "Network Energy vs Rounds",
                            num_dead_nodes, "Number of Dead Nodes",
                            "Number of Dead Nodes vs Rounds",
                            num_alive_nodes, "Number of Alive Nodes",
                            "Number of Alive Nodes vs Rounds")

        # Save the metrics dictionary to a file
        common.save_metrics(config=self.config,
                            network_energy=network_energy,
                            num_dead_nodes=num_dead_nodes,
                            num_alive_nodes=num_alive_nodes,
                            num_cluster_heads=num_cluster_heads,
                            pkt_delivery_ratio=pkt_delivery_ratio,
                            pkt_loss_ratio=pkt_loss_ratio,
                            control_packets_energy=control_packets_energy,
                            control_packet_bits=control_packet_bits,
                            pkts_sent_to_bs=pkts_sent_to_bs,
                            energy_dissipated=energy_dissipated)

    def evaluate_round(self, round):
        round += 1

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        self.num_cluster_heads = 0

        network_avg_energy = self.network.average_energy()

        potential_chs = self.potential_cluster_heads(network_avg_energy)

        self.choose_cluster_heads(potential_chs)
        self.network.create_clusters()
        self.net_model.dissipate_energy(round=round)

        return round

    def run_without_plotting(self, num_rounds, network_energy, num_dead_nodes,
                             num_alive_nodes, num_cluster_heads,
                             pkt_delivery_ratio, pkt_loss_ratio,
                             control_packets_energy, control_packet_bits,
                             pkts_sent_to_bs, energy_dissipated):
        round = 0
        with Progress() as progress:
            task = progress.add_task(
                "[red]Running LEACH_C...", total=num_rounds)
            while self.network.alive_nodes() > 0 and round < num_rounds:
                round = self.evaluate_round(round)

                common.add_to_metrics(self.config, self.network,
                                      round, network_energy,
                                      num_dead_nodes, num_alive_nodes,
                                      num_cluster_heads,
                                      pkt_delivery_ratio,
                                      pkt_loss_ratio,
                                      control_packets_energy,
                                      control_packet_bits,
                                      pkts_sent_to_bs,
                                      energy_dissipated)
                common.save_metrics(self.config, network_energy,
                                    num_dead_nodes, num_alive_nodes,
                                    num_cluster_heads,
                                    pkt_delivery_ratio,
                                    pkt_loss_ratio,
                                    control_packets_energy,
                                    control_packet_bits,
                                    pkts_sent_to_bs,
                                    energy_dissipated)
                progress.update(task, completed=round)
            progress.update(task, completed=num_rounds)

    def run_with_plotting(self, num_rounds, network_energy, num_dead_nodes,
                          num_alive_nodes, num_cluster_heads,
                          pkt_delivery_ratio, pkt_loss_ratio,
                          control_packets_energy, control_packet_bits,
                          pkts_sent_to_bs, energy_dissipated):
        fig, ax = plt.subplots()
        common.plot_clusters(network=self.network, round=0, ax=ax)

        def animate(round):
            round = self.evaluate_round(round)

            if round >= num_rounds or self.network.alive_nodes() <= 0:
                print("Done!")
                ani.event_source.stop()

            ax.clear()
            common.plot_clusters(network=self.network, round=round, ax=ax)

            common.add_to_metrics(self.config, self.network,
                                  round, network_energy,
                                  num_dead_nodes, num_alive_nodes,
                                  num_cluster_heads,
                                  pkt_delivery_ratio,
                                  pkt_loss_ratio,
                                  control_packets_energy,
                                  control_packet_bits,
                                  pkts_sent_to_bs,
                                  energy_dissipated)
            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
