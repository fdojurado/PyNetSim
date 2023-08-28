import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pynetsim.leach as leach


class LEACH_C:

    def __init__(self, network):
        self.name = "LEACH-C"
        self.config = network.config
        self.network = network
        self.elect = self.config.network.protocol.eelect_nano * 10**-9
        self.etx = self.config.network.protocol.etx_nano * 10**-9
        self.erx = self.config.network.protocol.erx_nano * 10**-9
        self.eamp = self.config.network.protocol.eamp_pico * 10**-12
        self.eda = self.config.network.protocol.eda_nano * 10**-9
        self.packet_size = self.config.network.protocol.packet_size

    def select_cluster_heads(self, network_avg_energy):
        potential_cluster_heads = []
        # Elect cluster heads
        for node in self.network.nodes.values():
            if leach.should_skip_node(node):
                continue

            node.is_cluster_head = False
            node.cluster_id = 0

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

        for node in self.network.nodes.values():
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
        initial_solution = np.random.choice(
            cluster_heads, size=len(cluster_heads), replace=False)

        # print(
        #     f"Initial solution: {initial_solution}, length: {len(initial_solution)}")

        # Evaluate the initial solution
        initial_solution_value = self.objective_function(initial_solution)

        # current working solution
        curr, curr_value = initial_solution, initial_solution_value

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
                curr = np.delete(curr, np.where(curr == cluster_head))
            else:
                # If the cluster head is not in the current solution, then add it
                curr = np.append(curr, cluster_head)

            # Evaluate the current solution
            curr_value = self.objective_function(curr)

            # If the current solution is better than the initial solution, then
            # accept the current solution
            if curr_value < initial_solution_value:
                initial_solution, initial_solution_value = curr, curr_value
            # Report progress
            # print(
            #     f"Best solution: {initial_solution}, length: {len(initial_solution)}")

            diff = initial_solution_value - curr_value

            t = initial_temp / (i + 1)
            metropolis = np.exp(-diff / t)
            if diff > 0 or np.random.rand() < metropolis:
                initial_solution, initial_solution_value = curr, curr_value

        # print(
        #     f"Best solution: {initial_solution}, length: {len(initial_solution)}")

        # Assign the cluster heads
        for node in self.network.nodes.values():
            if node.node_id in initial_solution:
                self.num_cluster_heads = leach.mark_as_cluster_head(
                    self.network, node, self.num_cluster_heads)

    def choose_cluster_heads(self, chs):
        # Use simulated annealing to choose the cluster heads
        self.simulated_annealing(chs)

    def run(self):
        print("Running LEACH_C...")
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = False

        for node in self.network.nodes.values():
            node.is_cluster_head = False

        network_energy = {}
        num_dead_nodes = {}
        num_alive_nodes = {}

        # Set all dst_to_sink for all nodes
        for node in self.network.nodes.values():
            node.dst_to_sink = self.network.distance_to_sink(node)

        if not plot_clusters_flag:
            self.run_without_plotting(
                num_rounds, network_energy, num_dead_nodes, num_alive_nodes)
        else:
            self.run_with_plotting(
                num_rounds, network_energy, num_dead_nodes, num_alive_nodes)

        leach.plot_metrics(network_energy, "Network Energy", "J",
                           "Network Energy vs Rounds",
                           num_dead_nodes, "Number of Dead Nodes",
                           "Number of Dead Nodes vs Rounds",
                           num_alive_nodes, "Number of Alive Nodes",
                           "Number of Alive Nodes vs Rounds")

        # Save the metrics dictionary to a file
        leach.save_metrics(config=self.config,
                           name=self.name,
                           network_energy=network_energy,
                           num_dead_nodes=num_dead_nodes,
                           num_alive_nodes=num_alive_nodes)

    def run_without_plotting(self, num_rounds, network_energy, num_dead_nodes,
                             num_alive_nodes):
        round = 0
        while self.network.alive_nodes() > 0 and round < num_rounds:
            round += 1
            print(f"Round {round}")

            # Clear all CHs from the previous round
            for node in self.network.nodes.values():
                node.is_cluster_head = False
                node.cluster_id = 0

            self.num_cluster_heads = 0

            # Compute the network average energy
            network_avg_energy = self.network.remaining_energy() / self.network.alive_nodes()

            # Select cluster heads
            chs = self.select_cluster_heads(network_avg_energy)
            # input("Press Enter to continue...")
            # Choose cluster heads
            self.choose_cluster_heads(chs)
            leach.create_clusters(network=self.network)
            leach.energy_dissipation_non_cluster_heads(round=round,
                                                       network=self.network,
                                                       elect=self.elect,
                                                       eda=self.eda,
                                                       packet_size=self.packet_size,
                                                       eamp=self.eamp)
            leach.energy_dissipation_cluster_heads(round=round,
                                                   network=self.network,
                                                   elect=self.elect,
                                                   eda=self.eda,
                                                   packet_size=self.packet_size,
                                                   eamp=self.eamp)

            leach.store_metrics(self.config, self.network,
                                round, network_energy,
                                num_dead_nodes, num_alive_nodes)
            leach.save_metrics(config=self.config,
                               name=self.name,
                               network_energy=network_energy,
                               num_dead_nodes=num_dead_nodes,
                               num_alive_nodes=num_alive_nodes)

    def run_with_plotting(self, num_rounds, network_energy, num_dead_nodes,
                          num_alive_nodes):
        fig, ax = plt.subplots()
        leach.plot_clusters(network=self.network, round=0, ax=ax)

        def animate(round):
            print(f"Round {round}")

            # Clear all CHs from the previous round
            for node in self.network.nodes.values():
                node.is_cluster_head = False
                node.cluster_id = 0

            self.num_cluster_heads = 0

            # Compute the network average energy
            network_avg_energy = self.network.remaining_energy() / self.network.alive_nodes()

            # Select cluster heads
            chs = self.select_cluster_heads(network_avg_energy)
            # Choose cluster heads
            self.choose_cluster_heads(chs)
            leach.create_clusters(network=self.network)
            leach.energy_dissipation_non_cluster_heads(round=round,
                                                       network=self.network,
                                                       elect=self.elect,
                                                       eda=self.eda,
                                                       packet_size=self.packet_size,
                                                       eamp=self.eamp)
            leach.energy_dissipation_cluster_heads(round=round,
                                                   network=self.network,
                                                   elect=self.elect,
                                                   eda=self.eda,
                                                   packet_size=self.packet_size,
                                                   eamp=self.eamp)

            ax.clear()
            leach.plot_clusters(network=self.network, round=round, ax=ax)

            leach.store_metrics(self.config, self.network,
                                round, network_energy,
                                num_dead_nodes, num_alive_nodes)

            if self.network.alive_nodes() <= 0:
                ani.event_source.stop()

            # Update the round number
            round += 1

            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
