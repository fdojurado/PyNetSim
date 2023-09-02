import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pynetsim.leach as leach

from rich.progress import Progress


class LEACH:

    def __init__(self, network):
        self.name = "LEACH"
        self.config = network.config
        self.network = network
        self.elect, self.etx, self.erx, self.eamp, self.eda, self.packet_size = leach.get_energy_conversion_factors(
            self.config)

    def select_cluster_heads(self, probability, tleft, num_cluster_heads):
        for node in self.network.nodes.values():
            if leach.should_skip_node(node):
                continue

            if node.rounds_to_become_cluster_head > 0:
                node.rounds_to_become_cluster_head -= 1

            node.is_cluster_head = False
            node.cluster_id = 0

            if self.should_select_cluster_head(node, probability):
                num_cluster_heads = self.mark_as_cluster_head(
                    node, num_cluster_heads, tleft)

        cluster_heads = [
            node.node_id for node in self.network.nodes.values() if node.is_cluster_head]
        # print(f"Cluster heads: {cluster_heads}")

    def should_select_cluster_head(self, node, probability):
        return (not node.is_cluster_head and node.rounds_to_become_cluster_head == 0
                and random.random() < probability)

    def mark_as_cluster_head(self, node, num_cluster_heads, tleft):
        num_cluster_heads = leach.mark_as_cluster_head(
            node, num_cluster_heads)
        node.rounds_to_become_cluster_head = 1 / \
            self.config.network.protocol.cluster_head_percentage - tleft
        # print(f"Node {node.node_id} is a cluster head.")
        return num_cluster_heads

    def run(self):
        print("Running LEACH protocol...")
        p = self.config.network.protocol.cluster_head_percentage
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = False  # Set this to False to not plot clusters

        for node in self.network.nodes.values():
            node.is_cluster_head = False

        network_energy = {}
        num_dead_nodes = {}
        num_alive_nodes = {}
        num_cluster_heads = {}

        # Set all dst_to_sink for all nodes
        for node in self.network.nodes.values():
            node.dst_to_sink = self.network.distance_to_sink(node)

        if not plot_clusters_flag:
            self.run_without_plotting(
                num_rounds, p, network_energy, num_dead_nodes, num_alive_nodes, num_cluster_heads)
        else:
            self.run_with_plotting(
                num_rounds, p, network_energy, num_dead_nodes, num_alive_nodes, num_cluster_heads)

        leach.plot_metrics(network_energy, "Network Energy", "J",
                           "Network Energy vs Rounds",
                           num_dead_nodes, "Number of Dead Nodes",
                           "Number of Dead Nodes vs Rounds",
                           num_alive_nodes, "Number of Alive Nodes",
                           "Number of Alive Nodes vs Rounds")
        leach.save_metrics(
            config=self.config,
            name=self.name,
            network_energy=network_energy,
            num_dead_nodes=num_dead_nodes,
            num_alive_nodes=num_alive_nodes,
            num_cluster_heads=num_cluster_heads
        )

    def evaluate_round(self, p, round):
        round += 1
        # print(f"Round {round}")

        num_cluster_heads = 0
        th = p / (1 - p * (round % (1 / p)))
        # print(f"Threshold: {th}")
        tleft = round % (1 / p)

        self.select_cluster_heads(th, tleft, num_cluster_heads)
        chs_bool = leach.create_clusters(network=self.network)
        leach.dissipate_energy(round=round, network=self.network,
                               elect=self.elect, eda=self.eda,
                               packet_size=self.packet_size, eamp=self.eamp)

        return round

    def run_without_plotting(self, num_rounds, p, network_energy, num_dead_nodes, num_alive_nodes,
                             num_cluster_heads_metric):
        round = 0
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Simulation Progress", total=num_rounds)

            while self.network.alive_nodes() > 0 and round < num_rounds:
                round = self.evaluate_round(p, round)

                leach.store_metrics(config=self.config, network=self.network, round=round,
                                    network_energy=network_energy,
                                    num_dead_nodes=num_dead_nodes,
                                    num_alive_nodes=num_alive_nodes,
                                    num_cluster_heads=num_cluster_heads_metric)
                leach.save_metrics(
                    config=self.config,
                    name=self.name,
                    network_energy=network_energy,
                    num_dead_nodes=num_dead_nodes,
                    num_alive_nodes=num_alive_nodes,
                    num_cluster_heads=num_cluster_heads_metric
                )
                progress.update(task, completed=round)
            progress.update(task, completed=num_rounds)

    def run_with_plotting(self, num_rounds, p, network_energy, num_dead_nodes, num_alive_nodes,
                          num_cluster_heads_metric):
        fig, ax = plt.subplots()
        leach.plot_clusters(network=self.network, round=0, ax=ax)

        def animate(round):
            round = self.evaluate_round(p, round)

            if round >= num_rounds or self.network.alive_nodes() <= 0:
                # print("Done!")
                ani.event_source.stop()

            ax.clear()
            leach.plot_clusters(network=self.network, round=round, ax=ax)

            leach.store_metrics(config=self.config, network=self.network, round=round,
                                network_energy=network_energy,
                                num_dead_nodes=num_dead_nodes,
                                num_alive_nodes=num_alive_nodes,
                                num_cluster_heads=num_cluster_heads_metric)
            leach.save_metrics(
                config=self.config,
                name=self.name,
                network_energy=network_energy,
                num_dead_nodes=num_dead_nodes,
                num_alive_nodes=num_alive_nodes,
                num_cluster_heads=num_cluster_heads_metric
            )

            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
