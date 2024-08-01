# Mixed-Integer Linear Programming (MILP) problem

import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyomo.environ as pyo
import copy

# from rich.progress import Progress


class LEACH_CE:

    def __init__(self, network, net_model: object):
        self.name = "LEACH-CE"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        # Define parameters
        self.alpha = 1.0  # Weighting coefficient for energy consumption
        self.beta = 1.0   # Weighting coefficient for balanced cluster energy

    def copy_network(self, network, net_model):
        network_copy = copy.deepcopy(network)
        net_model_copy = copy.deepcopy(net_model)
        net_model_copy.set_network(network_copy)
        return network_copy, net_model_copy

    def objective_function(self, model: object, network_copy: object, net_model_copy: object, round: int):
        print("Calculating objective function...")
        # Mark nodes as cluster heads those nodes that have x_i = 1
        for node in network_copy:
            if node.node_id == 1:
                continue
            # Check if the value of x_i is initialized
            if model.x[node.node_id].value is None:
                continue
            if pyo.value(model.x[node.node_id]) == 1:
                print(f"Node {node.node_id} is a cluster head.")
                # Get the cluster id, which is where the mode.y[node.node_id, cluster_id] = 1
                # Loop thorugh all values of the model.y[node.node_id] dictionary
                for cluster in model.y:
                    # get the node and cluster
                    node_id, cluster_id = cluster
                    # check if the node_id is the same as the node.node_id
                    if node_id != node.node_id:
                        continue
                    # check if the value of y_ij is different from 1
                    if pyo.value(model.y[node_id, cluster_id]) != 1:
                        continue
                    # if the value of y_ij is 1, then mark the node as a cluster head
                    print(f"Cluster id: {cluster_id}")
                    network_copy.mark_as_cluster_head(node, cluster_id)
            else:
                # Set the cluster ids for each non-cluster head node, which
                # is in the model.y dictionary
                for cluster in model.y[node.node_id]:
                    if pyo.value(model.y[node.node_id, cluster]) == 1:
                        node.cluster_id = cluster
                        break
        # print all cluster heads
        cluster_heads = [
            node.node_id for node in network_copy if node.is_cluster_head]
        print(f"Cluster heads: {cluster_heads}")
        # Get the current network remaining energy
        current_energy = network_copy.remaining_energy()
        print(f"Current energy: {current_energy}")
        net_model_copy.dissipate_energy(round=round)
        # Get the new network remaining energy
        new_energy = network_copy.remaining_energy()
        print(f"New energy: {new_energy}")
        energy = current_energy - new_energy
        print(f"Energy: {energy}")
        # Get the variance of the clusters
        variance = network_copy.get_clusters_variance_energy()
        print(f"Variance: {variance}")
        # Cost function
        cost = self.alpha * energy + self.beta * variance
        print(f"Cost: {cost}")
        return cost

    def energy_constraint_rule(self, node_id: int, network_copy: object, model: object):
        print(f"Calculating energy constraint for node {node_id}...")
        print(f"Network average remaining energy: {self.threshold_energy}")
        print(
            f"Node {node_id} remaining energy: {network_copy.get_node(node_id).remaining_energy}")
        x = model.x[node_id]
        node = network_copy.get_node(x)
        threshold_expr = (node.remaining_energy - self.threshold_energy)
        print(f"Threshold expr: {threshold_expr}")
        return threshold_expr >= 0

    def calculate_energy(self, node_id: int, network_copy: object):
        # print(f"Calculating energy for node {node_id}...")
        node = network_copy.get_node(node_id)
        return node.remaining_energy

    def choose_cluster_heads(self, round: int):
        # Decision variables
        # x_i: Binary variable that equals 1 if node i is selected as a cluster head, and 0 otherwise
        # y_ij: Binary variable that equals 1 if node i is assigned to cluster j, and 0 otherwise.
        # Create a copy of the network and the net_model
        network_copy, net_model_copy = self.copy_network(
            self.network, self.net_model)

        nodes = [node.node_id for node in network_copy if network_copy.should_skip_node(
            node) is False]
        print(f"Nodes alive for this round: {nodes}")
        # print potential chs
        print("No potential CHs: ", end="")
        for node in nodes:
            node_obj = network_copy.get_node(node)
            if node_obj.remaining_energy < self.threshold_energy:
                print(f"{node} ", end="")
        print()
        # Create a clusters list that goes from 1 to the self.max_chs
        clusters = [cluster for cluster in range(1, int(self.max_chs) + 1)]
        print(f"Clusters: {clusters}", len(clusters))
        model = pyo.ConcreteModel()
        model.x = pyo.Var(nodes, domain=pyo.Binary)
        # for node in nodes:  # Assuming nodes is the index set for model.x
        #     print(f"Node {node}: {model.x[node]}")
        model.y = pyo.Var(nodes, clusters, domain=pyo.Binary)
        # print the clusters
        # for node in nodes:
        #     for cluster in clusters:
        #         print(
        #             f"Node {node}, cluster {cluster}: {model.y[node, cluster]}")
        print("hello1")

        # Initial solution
        # Select randomly from the cluster heads without repeating
        number_of_cluster_heads = len(clusters)-2
        print(f"Number of cluster heads: {number_of_cluster_heads}")
        # Select randomly from the nodes
        selected_nodes = np.random.choice(
            nodes, size=number_of_cluster_heads, replace=False)
        print(f"Selected nodes: {selected_nodes}")
        # Set the x_i variables to 1 for the selected nodes
        for node in selected_nodes:
            model.x[node] = 1
        # Set the y_ij variables to 1 for the selected nodes
        visited_nodes = []
        for cluster in clusters:
            for node in selected_nodes:
                if node in visited_nodes:
                    continue
                print(f"Setting y[{node}, {cluster}] to 1")
                model.y[node, cluster] = 1
                # Get the other clusters
                other_clusters = [c for c in clusters if c != cluster]
                # Set the y_ij variables to 0 for the other clusters
                for c in other_clusters:
                    print(f"Setting y[{node}, {c}] to 0")
                    model.y[node, c] = 0
                visited_nodes.append(node)
                break

        # Objective function
        model.OBJ = pyo.Objective(
            rule=self.objective_function(
                model, network_copy, net_model_copy, round),
            sense=pyo.minimize)

        print("hello2")

        # Create a constraint for each node
        model.energy_constraints = pyo.ConstraintList()
        for node in nodes:
            # Get the node
            node_obj = network_copy.get_node(node)
            # Get the node's remaining energy
            node_remaining_energy = node_obj.remaining_energy
            # Create the constraint
            model.energy_constraints.add(
                expr=model.x[node]*node_remaining_energy >= self.threshold_energy)

        # Create constrains such that each node that is assigned to a cluster has to have enough energy to transmit to the cluster head
        model.cluster_energy_constraints = pyo.ConstraintList()
        for node in nodes:
            # Get the node
            node_obj = network_copy.get_node(node)
            # Get the node's remaining energy
            node_remaining_energy = node_obj.remaining_energy
            for cluster in clusters:
                # Create the constraint
                model.cluster_energy_constraints.add(
                    expr=model.y[node, cluster]*node_remaining_energy >= 0)

        # Create a constraint for each node such that each node can only be assigned to one cluster
        model.cluster_assignment_constraint = pyo.ConstraintList()
        for node in nodes:
            model.cluster_assignment_constraint.add(
                expr=sum(model.y[node, cluster] for cluster in clusters) == 1)

        # Create a constraint such that the number of cluster heads is equal to the self.max_chs
        model.cluster_head_selection_constraint = pyo.Constraint(
            expr=sum(model.x[node] for node in nodes) == self.max_chs)

        # Solve the problem
        solver = pyo.SolverFactory('glpk')
        # solver.options['write'] = '/Users/fernando/PyNetSim/tutorials/glpk/tmp12odqxon.glpk.raw'
        # solver.options['wglp'] = '/Users/fernando/PyNetSim/tutorials/glpk/tmp2bbdre7d.glpk.glp'
        # solver.options['cpxlp'] = '/Users/fernando/PyNetSim/tutorials/glpk/tmphrqkh3tl.pyomo.lp'
        results = solver.solve(model, tee=True)
        # Check the status of the optimization
        chs = []
        # Check the solver status
        if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Access the optimal objective value
            optimal_objective_value = pyo.value(model.OBJ)
            print(f"Optimal Objective Value: {optimal_objective_value}")
            for node in nodes:
                if pyo.value(model.x[node]) == 1:
                    chs.append(node)
        else:
            print("No optimal solution found.")

        # Display the objective function value
        print(f"Objective Function Value: {model.OBJ()}")
        return chs

    def run(self):
        print("Running LEACH-CE...")
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

    def update_cluster_heads(self, chs):
        for node in self.network:
            if node.node_id in chs:
                self.network.mark_as_cluster_head(
                    node, node.node_id)

    def evaluate_round(self, round):
        round += 1

        for node in self.network:
            self.network.mark_as_non_cluster_head(node)

        self.threshold_energy = self.network.average_remaining_energy()

        print(f"Threshold energy: {self.threshold_energy}")

        self.max_chs = np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage)

        print(f"Max CHs: {self.max_chs}")

        chs = self.choose_cluster_heads(round=round)
        print(f"Cluster heads at round {round}: {chs}")
        self.update_cluster_heads(chs)
        self.network.create_clusters()
        self.net_model.dissipate_energy(round=round)
        input("Press Enter to continue...")

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
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
