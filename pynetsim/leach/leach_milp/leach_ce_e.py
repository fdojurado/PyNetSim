# Mixed-Integer Linear Programming (MILP) problem
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import copy
import pynetsim.leach.leach_milp as leach_milp
import matplotlib.animation as animation

from pynetsim import common
from decimal import Decimal as D


class LEACH_CE_E:
    def __init__(self, network, net_model):
        self.name = "LEACH-CE-E"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.alpha = 2
        self.beta = 0.7
        self.gamma = 2
        self.a = 4
        self.b = 0.5

    @staticmethod
    def create_model(network: object, cluster_heads: list, alive_nodes: list, a: int, b: int,
                     alpha: float, beta: float, gamma: float, max_chs: int):
        model = pyo.ConcreteModel()

        # Decision variables
        model.x = pyo.Var(cluster_heads, within=pyo.Binary)
        model.y = pyo.Var(alive_nodes, cluster_heads, within=pyo.Binary)

        model.cluster_heads = pyo.Set(initialize=cluster_heads)
        model.nodes = pyo.Set(initialize=alive_nodes)

        # Parameter representing the energy spent by a non-cluster head node to transmit to a cluster head
        energy_spent_non_ch = {i: {j: leach_milp.energy_spent_non_ch(
            network, i, j) for j in model.cluster_heads if j != i} for i in model.nodes}

        model.energy_spent_non_ch = pyo.Param(
            model.nodes, initialize=lambda model, node: energy_spent_non_ch[node], mutable=False)

        # Parameter representing the energy spent by a cluster head node to transmit to the sink
        model.energy_spent_ch = pyo.Param(
            model.cluster_heads,
            initialize=lambda model, node: leach_milp.energy_spent_ch(
                network, node),
            mutable=False)

        # Parameter representing the current remaining energy of each node
        model.remaining_energy = pyo.Param(
            model.nodes,
            initialize=lambda model, node: leach_milp.get_energy(
                network.get_node(node)),
            mutable=False)

        remaining_energy = {}
        for node in model.remaining_energy:
            if node in model.cluster_heads:
                remaining_energy[node] = model.remaining_energy[node]

        # sort and print
        remaining_energy = {k: v for k, v in sorted(
            remaining_energy.items(), key=lambda item: item[1], reverse=True)}
        # print(f"Remaining energy: {remaining_energy}")

        # Parameter representing the target load balancing for each cluster head
        model.target_load_balancing = pyo.Param(
            model.cluster_heads,
            initialize=lambda model, node: leach_milp.target_load_balancing(
                network, node, a, b),
            mutable=False)

        model.abs_load_balancing = pyo.Var()

        # print model.y
        # for y in model.y:
        #     input(f"y[{y}] = {model.y[y]}")

        # print model.[2,x]
        # for x in model.x:
        #     input(f"model.y[2,x] = {model.y[2,x]}")

        load_balancing = 0
        for cluster_head in model.cluster_heads:
            load_balancing += a * \
                model.remaining_energy[cluster_head] * model.x[cluster_head]
            num_nodes = sum(model.y[node, cluster_head]
                            for node in model.nodes)
            load_balancing += b * num_nodes
            load_balancing -= model.target_load_balancing[cluster_head] * \
                model.x[cluster_head]

        # load_balancing = sum(self.a*model.remaining_energy[node]*model.x[node] for node in model.cluster_heads) + \
            # sum(self.b*model.y[node, cluster_head] for node in model.nodes for cluster_head in model.cluster_heads if cluster_head != node) + \
            # sum(-1*model.target_load_balancing[node]*model.x[node]
            #     for node in model.cluster_heads)
        # sum(self.b*model.y[node][cluster_head] for node in model.nodes for cluster_head in model.cluster_heads if cluster_head != node) + \

        # input(f"Load balancing: {load_balancing}")

        model.abs_value_constraint = pyo.Constraint(
            expr=model.abs_load_balancing >= load_balancing,
        )

        model.abs_value_constraint2 = pyo.Constraint(
            expr=model.abs_load_balancing >= -load_balancing,
        )
        # # Objective function
        model.OBJ = pyo.Objective(
            sense=pyo.minimize, rule=lambda model: LEACH_CE_E.objective_function(model, alpha, beta, gamma))

        # print objective expression
        # input(f"Objective expression: {model.OBJ.expr}")

        def assignment_rule(model, i):
            return sum(model.y[i, j] for j in model.cluster_heads) == 1

        # Ensure that each node is assigned to one cluster
        model.node_cluster = pyo.Constraint(
            model.nodes, rule=assignment_rule)

        def cluster_heads_limit_rule(model):
            return sum(model.x[j] for j in model.cluster_heads) == max_chs

        # Ensure that the number of selected cluster heads does not exceed a predefined limit
        model.cluster_heads_limit = pyo.Constraint(
            rule=cluster_heads_limit_rule)

        def consistency_rule(model, i, j):
            return model.y[i, j] <= model.x[j]

        # Constraint: Ensure y_ij is consistent with x_j
        model.consistency_constraint = pyo.Constraint(
            model.nodes, model.cluster_heads, rule=consistency_rule)

        # Constraint to limit the number of non-cluster head nodes per cluster
        def non_cluster_head_limit_rule(model, j):
            return sum(model.y[i, j] for i in model.nodes) <= np.ceil(1.5*(len(alive_nodes) / max_chs))

        model.non_cluster_head_limit = pyo.Constraint(
            model.cluster_heads, rule=non_cluster_head_limit_rule)

        return model

    @staticmethod
    def objective_function(model: object, alpha: float, beta: float, gamma: float):
        # Sum the distances from each node to its cluster head
        # expr = self.alpha * sum(model.distances[node][cluster_head] * model.y[node, cluster_head]
        #                         for node in model.nodes for cluster_head in model.cluster_heads if cluster_head != node)
        # # Now, subtract the distances from each potential cluster head that is selected to be a cluster head to all other cluster heads
        # expr += self.alpha * sum(-1*model.distances[node][cluster_head] * model.x[node]
        #                         for node in model.cluster_heads for cluster_head in model.cluster_heads if cluster_head != node)
        # # Now, add the load balancing term
        # expr += self.gamma * model.abs_load_balancing
        # return expr
        # Sum the energy spent from from each node to its cluster head
        expr = alpha * sum(model.energy_spent_non_ch[node][cluster_head] * model.y[node, cluster_head]
                           for node in model.nodes for cluster_head in model.cluster_heads if cluster_head != node)
        #  Now, subtract the energy spent from each potential cluster head that is selected to be a cluster head to all other cluster heads
        # expr += self.alpha * sum(-1*model.energy_spent_non_ch[node][cluster_head] * model.x[node]
        #                          for node in model.cluster_heads for cluster_head in model.cluster_heads if cluster_head != node)
        # Now, add the energy spent from each cluster head to the sink
        expr += beta * sum(model.energy_spent_ch[node] * model.x[node]
                           for node in model.cluster_heads)
        # Now, add the load balancing term
        expr += gamma * model.abs_load_balancing
        return expr
        # return self.alpha * sum(model.energy_spent_non_ch[node][other_node] * model.y[node, other_node]
        #                         for node in model.nodes for other_node in model.cluster_heads if other_node != node) + \
        #     self.beta*sum(model.energy_spent_ch[node] * model.x[node]
        #                   for node in model.cluster_heads) + self.gamma * model.abs_load_balancing

    @staticmethod
    def choose_cluster_heads(network: object, threshold_energy: float,
                             alpha: float = 2, beta: float = 0.7, gamma: float = 2, a: int = 4, b: int = 0.5, max_chs: int = 2):
        alive_nodes = [node.node_id for node in network if not network.should_skip_node(
            node) and network.alive(node)]

        # Potential cluster heads
        cluster_heads = [node for node in alive_nodes if network.get_node(
            node).remaining_energy >= threshold_energy]

        # print(f"Potential cluster heads: {cluster_heads}")

        # print("Alive nodes IDs: ", end="")
        # for node in alive_nodes:
        #     print(node, end=" ")
        # print()

        if len(cluster_heads) == 0 and len(alive_nodes) == 1:
            cluster_heads = alive_nodes

        model = LEACH_CE_E.create_model(
            network=network, cluster_heads=cluster_heads, alive_nodes=alive_nodes, a=a, b=b, alpha=alpha, beta=beta, gamma=gamma, max_chs=max_chs)

        # Solve the problem
        solver = pyo.SolverFactory('glpk', tee=True)
        solver.options['tmlim'] = 60
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
            # self.print_model(model)
        else:
            print("No solution found.")
            # raise Exception("No solution found.")
            raise Exception("No solution found.")

        # Display the objective function value
        # input(f"Objective Function Value: {model.OBJ()}")
        return chs, node_cluster_head

    def print_model(self, model):
        chs = []
        # print the optimal solution
        for node in model.cluster_heads:
            if pyo.value(model.x[node]) == 1:
                chs.append(node)

        print(f"Cluster heads: {chs}")

        # dict that holds the current energy level of the cluster heads
        chs_energy = {}
        for node in chs:
            chs_energy[node] = leach_milp.get_energy(
                self.network.get_node(node))

        # Sort the cluster heads by their energy level in descending order
        chs_energy = {k: v for k, v in sorted(
            chs_energy.items(), key=lambda item: item[1], reverse=True)}

        print(f"Cluster heads sorted by energy: {chs_energy}")

        # print nodes assigned to each cluster head, key is the cluster head, value is the list of nodes assigned to it
        clusters_assignment = {}
        for node in model.nodes:
            for cluster_head in model.cluster_heads:
                if pyo.value(model.y[node, cluster_head]) == 1:
                    if cluster_head not in clusters_assignment:
                        clusters_assignment[cluster_head] = []
                    clusters_assignment[cluster_head].append(node)

        print(f"Clusters assignment: {clusters_assignment}")

        # How many nodes are assigned to each cluster head
        for cluster_head in clusters_assignment:
            print(
                f"Cluster head {cluster_head} has {len(clusters_assignment[cluster_head])} nodes assigned to it.")
        input("Press Enter to continue...")

    def run(self):
        print(f"Running {self.name}...")
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

        self.threshold_energy = D(f"{self.network.average_remaining_energy()}")

        print(f"Threshold energy: {self.threshold_energy}")

        self.max_chs = np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage)

        print(f"Max CHs: {self.max_chs}")

        chs, node_cluster_head = LEACH_CE_E.choose_cluster_heads(
            network=self.network, threshold_energy=self.threshold_energy, max_chs=self.max_chs, a=self.a, b=self.b, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
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
