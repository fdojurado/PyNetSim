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

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import copy
import pynetsim.leach.leach_milp as leach_milp

import matplotlib.animation as animation

from pynetsim import common
from rich.progress import Progress


class LEACH_CE_E:
    def __init__(self, network, net_model, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = "LEACH-CE-E"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 54.82876630831832
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        else:
            self.beta = 14.53707859358856
        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        else:
            self.gamma = 35.31010127750784
        self.network.set_stats_name(
            f'{self.name}_{self.alpha}_{self.beta}_{self.gamma}')

    def create_model(self, cluster_heads, alive_nodes):
        model = pyo.ConcreteModel()

        # Decision variables
        model.x = pyo.Var(cluster_heads, within=pyo.Binary)
        model.y = pyo.Var(alive_nodes, cluster_heads, within=pyo.Binary)
        # model.abs_load_balancing = pyo.Var()

        model.cluster_heads = pyo.Set(initialize=cluster_heads)
        model.nodes = pyo.Set(initialize=alive_nodes)

        # Parameter representing the distance between nodes
        distances = {i: {j: self.network.distance_between_nodes(
            self.network.get_node(i), self.network.get_node(j)) for j in model.nodes} for i in model.nodes}

        model.distances = pyo.Param(
            model.nodes, model.nodes, initialize=lambda model, i, j: distances[i][j], mutable=False)

        # Energy spent by a node when transmitting to a cluster head
        energy_spent_non_ch = {i: {j: leach_milp.energy_spent_non_ch(
            self.network, i, j) for j in model.cluster_heads} for i in model.nodes}

        model.energy_spent_non_ch = pyo.Param(
            model.nodes, model.cluster_heads, initialize=lambda model, i, j: energy_spent_non_ch[i][j], mutable=False)

        # Energy spent by a cluster head when transmitting to the sink
        energy_spent_ch_tx = {i: leach_milp.energy_spent_ch(
            self.network, i) for i in model.cluster_heads}

        model.energy_spent_ch_tx = pyo.Param(
            model.cluster_heads, initialize=lambda model, i: energy_spent_ch_tx[i], mutable=False)

        model.energy_spent_ch_rx_per_node = pyo.Param(
            initialize=leach_milp.calculate_energy_ch_rx_per_node(self.network), mutable=False)

        # Objective function
        model.OBJ = pyo.Objective(
            sense=pyo.minimize, rule=self.objective_function)

        def assignment_rule(model, i):
            return sum(model.y[i, j] for j in model.cluster_heads) == 1

        # Ensure that each node is assigned to one cluster
        model.node_cluster = pyo.Constraint(
            model.nodes, rule=assignment_rule)

        def cluster_heads_limit_rule(model):
            return sum(model.x[j] for j in model.cluster_heads) == self.max_chs

        # Ensure that the number of selected cluster heads does not exceed a predefined limit
        model.cluster_heads_limit = pyo.Constraint(
            rule=cluster_heads_limit_rule)

        def consistency_rule(model, i, j):
            return model.y[i, j] <= model.x[j]

        # Constraint: Ensure y_ij is consistent with x_j
        model.consistency_constraint = pyo.Constraint(
            model.nodes, model.cluster_heads, rule=consistency_rule)

        return model

    def objective_function(self, model):
        expr = self.alpha * sum(model.energy_spent_non_ch[node, other_node] * model.y[node, other_node]
                                for node in model.nodes for other_node in model.cluster_heads)

        expr += self.beta * sum(model.energy_spent_ch_tx[cluster_head] * model.x[cluster_head]
                                for cluster_head in model.cluster_heads)

        # Energy spent by a cluster head when receiving from a node
        for cluster_head in model.cluster_heads:
            num_nodes = sum(model.y[node, cluster_head]
                            for node in model.nodes)
            energy_spent_ch_rx = num_nodes * model.energy_spent_ch_rx_per_node
            expr += self.gamma * energy_spent_ch_rx

        return expr

    def choose_cluster_heads(self):
        alive_nodes = [node.node_id for node in self.network if not self.network.should_skip_node(
            node) and self.network.alive(node)]

        # Potential cluster heads
        cluster_heads = [node for node in alive_nodes if self.network.get_node(
            node).remaining_energy >= self.threshold_energy]

        model = self.create_model(cluster_heads, alive_nodes)

        # Solve the problem
        solver = pyo.SolverFactory('glpk')
        results = solver.solve(model)

        chs = []
        # cluster heads assignment
        node_cluster_head = {}
        if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Access the optimal objective value
            # optimal_objective_value = pyo.value(model.OBJ)
            # print(f"Optimal Objective Value: {optimal_objective_value}")
            # print(f"x: {model.x.pprint()}")
            # print the optimal solution
            for node in model.cluster_heads:
                if pyo.value(model.x[node]) == 1:
                    chs.append(node)
                    # print(f"Node {node} is a cluster head.")
            # print nodes assigned to each cluster head
            for node in model.nodes:
                for cluster_head in model.cluster_heads:
                    if pyo.value(model.y[node, cluster_head]) == 1:
                        node_cluster_head[node] = cluster_head
                        pass

        # If no cluster heads were selected, then select the self.max_chs node from the potential cluster heads with the highest energy
        if len(chs) == 0:
            # Select the self.max_chs node from the potential cluster heads with the highest energy
            sorted_nodes = sorted(
                cluster_heads, key=lambda node: self.network.get_node(node).remaining_energy, reverse=True)
            chs = sorted_nodes[:self.max_chs]
            # print(f"Selected cluster heads: {chs}, max_chs: {self.max_chs}, len(sorted_nodes): {len(sorted_nodes)}")
            # Now assign the nodes to the closest cluster head
            for node in model.nodes:
                min_distance = 10000
                ch_id = -1
                for ch in chs:
                    # Calculate the distance between the node and the cluster head
                    distance = self.network.distance_between_nodes(
                        self.network.get_node(node), self.network.get_node(ch))
                    if distance < min_distance:
                        min_distance = distance
                        ch_id = ch
                node_cluster_head[node] = ch_id

        # if len(chs) < 5:
        #     print(f"Length of cluster heads: {len(chs)}")
        #     # add 0 to the beginning of the list to make it 5 elements
        #     chs = [0]*(5-len(chs)) + chs
        #     print(f"chs: {chs}, len(chs): {len(chs)}")

        # Lets check if the nodes are assigned to a cluster head that is the closest to them
        # for node in model.nodes:
        #     if node not in chs:
        #         min_distance = 10000
        #         ch_distance = 10000
        #         ch_id = -1
        #         ch_assigned = -1
        #         for ch in chs:
        #             # Calculate the distance between the node and the cluster head
        #             distance = self.network.distance_between_nodes(
        #                 self.network.get_node(node), self.network.get_node(ch))
        #             if distance < min_distance:
        #                 min_distance = distance
        #                 ch_id = ch
        #             if pyo.value(model.y[node, ch]) == 1:
        #                 ch_distance = model.distances[node, ch]
        #                 ch_assigned = ch
        #                 break
        #         if ch_id != ch_assigned and min_distance != ch_distance:
        #             raise Exception(
        #                 f"Node {node} is not assigned to the closest cluster head. Assigned to {ch_assigned}, but closest is {ch_id}.")

        # Display the objective function value
        # input(f"Objective Function Value: {model.OBJ()}")
        return chs, node_cluster_head

    def run(self):
        print(
            f"Running {self.name} with alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}")
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = self.config.network.plot
        plot_refresh = self.config.network.plot_refresh

        for node in self.network:
            node.is_cluster_head = False

        # Set all dst_to_sink for all nodes
        for node in self.network:
            node.dst_to_sink = self.network.distance_to_sink(node)

        if not plot_clusters_flag:
            self.run_without_plotting(
                num_rounds)
        else:
            self.run_with_plotting(
                num_rounds, plot_refresh)
        # export the metrics
        self.network.export_stats()

    def evaluate_round(self, round):
        round += 1

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        self.threshold_energy = self.network.average_remaining_energy()

        # print(f"Threshold energy: {self.threshold_energy}")

        self.max_chs = int(np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage))

        # print(f"Max CHs: {self.max_chs}")

        chs, node_cluster_head = self.choose_cluster_heads()
        # print(f"Cluster heads at round {round}: {chs}")
        leach_milp.update_cluster_heads(self.network, chs)
        leach_milp.update_chs_to_nodes(self.network, node_cluster_head)
        self.net_model.dissipate_energy(round=round)
        # input("Press Enter to continue...")

        return round

    def run_without_plotting(self, num_rounds):
        round = 0
        with Progress() as progress:
            task = progress.add_task(
                f"[red]Running {self.name}...", total=num_rounds)
            while self.network.alive_nodes() > 0 and round < num_rounds:
                round = self.evaluate_round(round)
                progress.update(task, completed=round)
            progress.update(task, completed=num_rounds)

    def run_with_plotting(self, num_rounds, plot_refresh):
        fig, ax = plt.subplots()
        common.plot_clusters(network=self.network, round=0, ax=ax)

        def animate(round):
            round = self.evaluate_round(round)

            if round >= num_rounds or self.network.alive_nodes() <= 0:
                print("Done!")
                ani.event_source.stop()

            ax.clear()
            common.plot_clusters(network=self.network, round=round, ax=ax)

            plt.pause(plot_refresh)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(0, num_rounds + 1), repeat=False)

        plt.show()
