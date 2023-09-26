# Mixed-Integer Linear Programming (MILP) problem
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import copy
import pynetsim.leach.leach_milp as leach_milp

from pynetsim import common


class LEACH_CE_D:
    def __init__(self, network, net_model):
        self.name = "LEACH-CE-D"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.alpha = 1
        self.beta = 3
        self.gamma = 3

    def create_model(self, cluster_heads, alive_nodes):
        model = pyo.ConcreteModel()

        # Decision variables
        model.x = pyo.Var(cluster_heads, within=pyo.Binary)
        model.y = pyo.Var(alive_nodes, cluster_heads, within=pyo.Binary)
        model.z = pyo.Var(cluster_heads, cluster_heads, within=pyo.Binary)

        model.cluster_heads = pyo.Set(initialize=cluster_heads)
        model.nodes = pyo.Set(initialize=alive_nodes)

        # Parameter representing the distance from a node to all other nodes
        distances = {i: {j: leach_milp.dist_between_nodes(self.network,
                                                          i, j) for j in model.nodes} for i in model.nodes}

        # max distance
        model.max_distance = pyo.Param(
            initialize=max([distances[i][j] for i in model.nodes for j in model.nodes]))

        model.distances = pyo.Param(
            model.nodes, model.cluster_heads, initialize=lambda model, i, j: distances[i][j]/model.max_distance if i != j else 0, mutable=False)

        print("Alive nodes: ", alive_nodes)

        print(f"Max distance: {model.max_distance.value}")

        # Max remaining energy
        model.max_remaining_energy = pyo.Param(
            initialize=max([self.network.get_node(node).remaining_energy for node in model.nodes]))

        print(f"Max remaining energy: {model.max_remaining_energy.value}")

        # Parameter representing the current remaining energy of each node
        model.remaining_energy = pyo.Param(
            model.nodes,
            initialize=lambda model, node: self.network.get_node(
                node).remaining_energy/model.max_remaining_energy,
            mutable=False)

        # Objective function
        model.OBJ = pyo.Objective(
            sense=pyo.minimize, rule=self.objective_function)

        def assignment_rule(model, i):
            return sum(model.y[i, j] for j in model.cluster_heads) == 1

        # Ensure that each node is assigned to one cluster
        model.node_cluster = pyo.Constraint(
            model.nodes, rule=assignment_rule)

        def cluster_heads_limit_rule(model):
            return sum(model.x[j] for j in model.cluster_heads) <= self.max_chs

        # Ensure that the number of selected cluster heads does not exceed a predefined limit
        model.cluster_heads_limit = pyo.Constraint(
            rule=cluster_heads_limit_rule)

        def consistency_rule(model, i, j):
            return model.y[i, j] <= model.x[j]

        # Constraint: Ensure y_ij is consistent with x_j
        model.consistency_constraint = pyo.Constraint(
            model.nodes, model.cluster_heads, rule=consistency_rule)

        # Constraint: Model z_ij
        def z_rule(model, i, j):
            return model.z[i, j] >= model.x[i]+model.x[j]-1

        model.z_constraint = pyo.Constraint(
            model.cluster_heads, model.cluster_heads, rule=z_rule)

        # Constraint to limit the number of non-cluster head nodes per cluster
        def non_cluster_head_limit_rule(model, j):
            return sum(model.y[i, j] for i in model.nodes) <= np.ceil(1.2*(len(alive_nodes) / self.max_chs))

        model.non_cluster_head_limit = pyo.Constraint(
            model.cluster_heads, rule=non_cluster_head_limit_rule)

        # Constraint: Model z_ij model.z <= x_i
        # def z_rule2(model, i, j):
        #     return model.z[i, j] <= model.x[i]

        # model.z_constraint2 = pyo.Constraint(
        #     model.cluster_heads, model.cluster_heads, rule=z_rule2)

        # # Constraint: Model z_ij model.z <= x_j
        # def z_rule3(model, i, j):
        #     return model.z[i, j] <= model.x[j]

        # model.z_constraint3 = pyo.Constraint(
        #     model.cluster_heads, model.cluster_heads, rule=z_rule3)

        return model

    def objective_function(self, model):
        non_chs = sum(model.distances[node, ch] * model.y[node, ch]
                      for node in model.nodes for ch in model.cluster_heads)
        # Minimize the distance between cluster heads and the sink
        chs = sum(self.network.get_node(node).dst_to_sink *
                  model.x[node] for node in model.cluster_heads)
        # Now, lets maximize the distance between cluster heads
        # bt_chs = sum(-1*model.distances[node, ch] * model.z[node, ch]
        #              for node in model.cluster_heads for ch in model.cluster_heads)
        return self.alpha * non_chs + self.beta * chs
        # return self.alpha * sum(model.distances[node][other_node] * model.y[node, other_node]
        #                         for node in model.nodes for other_node in model.cluster_heads if other_node != node) - \
        #     self.beta * sum(model.distances[node][other_node] * model.x[node]
        #                     for node in model.cluster_heads for other_node in model.cluster_heads if other_node != node and model.x[other_node].value)

    def choose_cluster_heads(self, round):
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

        # Display the objective function value
        # input(f"Objective Function Value: {model.OBJ()}")
        return chs, node_cluster_head

    def run(self):
        print("Running LEACH-CE-D...")
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = False

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
                num_rounds)

    def evaluate_round(self, round):
        round += 1

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        self.threshold_energy = self.network.average_remaining_energy()

        print(f"Threshold energy: {self.threshold_energy}")

        self.max_chs = np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage)

        print(f"Max CHs: {self.max_chs}")

        chs, node_cluster_head = self.choose_cluster_heads(round)
        print(f"Cluster heads at round {round}: {chs}")
        leach_milp.update_cluster_heads(self.network, chs)
        leach_milp.update_chs_to_nodes(self.network, node_cluster_head)
        self.net_model.dissipate_energy(round=round)
        # input("Press Enter to continue...")

        return round

    def run_without_plotting(self, num_rounds):
        round = 0
        # with Progress() as progress:
        # task = progress.add_task(
        #     "[red]Running LEACH-CE...", total=num_rounds)
        while self.network.alive_nodes() > 0 and round < num_rounds:
            round = self.evaluate_round(round)
            #     progress.update(task, completed=round)
            # progress.update(task, completed=num_rounds)

    def run_with_plotting(self, num_rounds):
        fig, ax = plt.subplots()
        common.plot_clusters(network=self.network, round=0, ax=ax)

        def animate(round):
            round = self.evaluate_round(round)

            if round >= num_rounds or self.network.alive_nodes() <= 0:
                print("Done!")
                ani.event_source.stop()

            ax.clear()
            common.plot_clusters(network=self.network, round=round, ax=ax)

            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
