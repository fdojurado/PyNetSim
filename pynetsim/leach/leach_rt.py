import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyomo.environ as pyo
import pynetsim.leach.leach_milp as leach_milp
import json
import os
import errno


from pynetsim.leach.leach_milp.leach_ce_e import LEACH_CE_E
from rich.progress import Progress
from decimal import Decimal as D


class LEACH_RT:

    def __init__(self, network, net_model: object):
        self.name = "LEACH-RT"
        self.net_model = net_model
        self.elect = self.net_model.elect
        self.config = network.config
        self.network = network

    def calculate_energy_consumption_per_round(self, network: object = None, net_model: object = None):
        # Calculate the energy spent by non-cluster heads
        E_non_ch = 0
        for node in network:
            if network.should_skip_node(node) or node.is_cluster_head:
                continue
            # Get the cluster head of the node
            cluster_head = network.get_cluster_head(node)
            # Calculate the distance between the node and its cluster head
            distance = network.distance_between_nodes(node, cluster_head)
            # Calculate the energy spent by the node to transmit to its cluster head
            E_non_ch += net_model.calculate_energy_tx_non_ch(distance)
        # Calculate the energy spent by cluster heads
        E_ch = 0
        E_ch_rx = 0
        for node in network:
            if network.should_skip_node(node) or not node.is_cluster_head:
                continue
            # Calculate the energy spent by the node to transmit to the sink
            E_ch += net_model.calculate_energy_tx_ch(node.dst_to_sink)
            # Calculate the energy spent by the node to receive data from non-cluster heads
            E_ch_rx += net_model.calculate_energy_rx()
        # Calculate the energy spent by control packets
        E_control = 0
        # Get the number of cluster heads
        pkt_size = 400
        energy = self.elect * pkt_size
        for node in network:
            if network.should_skip_node(node):
                continue
            E_control += energy

        return E_non_ch + E_ch + E_ch_rx + E_control, E_control

    def solve_milp(self):
        # We first solve the MILP until the first node dies
        node_dead = False
        alive_nodes = self.network.alive_nodes()
        round = 0
        data = {}
        # Copy the network and network model
        network_copy, net_model_copy = leach_milp.copy_network(
            network=self.network, net_model=self.net_model)
        while not node_dead:
            self.threshold_energy = D(
                f"{network_copy.average_remaining_energy()}")
            self.max_chs = np.ceil(
                network_copy.alive_nodes() * self.config.network.protocol.cluster_head_percentage)
            # Call a function that generates the new set of clusters
            chs, non_chs = LEACH_CE_E.choose_cluster_heads(
                network=network_copy, threshold_energy=self.threshold_energy, max_chs=self.max_chs)
            leach_milp.update_cluster_heads(
                network=network_copy, chs=chs)
            # print cluster heads
            leach_milp.update_chs_to_nodes(
                network=network_copy, assignments=non_chs)
            net_model_copy.dissipate_energy(round=round)
            energy_per_round, ctrl_energy_per_round = self.calculate_energy_consumption_per_round(
                network=network_copy, net_model=net_model_copy)

            data[round] = {
                "chs": chs,
                "non_chs": non_chs,
                "energy_per_round": energy_per_round,
                "ctrl_energy_per_round": ctrl_energy_per_round
            }

            # Lets stop if there is one node dead
            if alive_nodes != network_copy.alive_nodes():
                node_dead = True

            # if network_copy.alive_nodes() <= 0:
            #     node_dead = True
            round += 1

        print(f"Data: {data}")
        return data

    def solve_optimal_r(self, data):
        # Now solve the LIP to find the optimal R
        model = pyo.ConcreteModel()

        nodes = [node.node_id for node in self.network if node.node_id != 1]

        last_round = 1000

        max_rounds = range(1, last_round+1)

        print(f"Nodes: {nodes}")

        # Decision variables
        # model.alive = pyo.Var(
        #     nodes, max_rounds, within=pyo.Binary, initialize=1)
        model.first_death_round = pyo.Var(
            max_rounds, within=pyo.Binary)
        model.cluster_heads = pyo.Var(nodes, max_rounds, within=pyo.Binary)
        model.y = pyo.Var(nodes, nodes, max_rounds, within=pyo.Binary)
        model.cluster_head_change = pyo.Var(max_rounds, within=pyo.Binary)

        # Define the set of rounds
        model.rounds = pyo.RangeSet(last_round)
        model.nodes = pyo.Set(initialize=nodes)

        # Parameter representing the distance from a node to all other nodes including the sink
        nodes.append(1)
        nodes.sort()
        distance = {i: {j: leach_milp.dist_between_nodes(self.network, i, j) for j in nodes}
                    for i in model.nodes}

        model.distance = pyo.Param(
            model.nodes, nodes,
            initialize=lambda model, i, j: distance[i][j],
            mutable=False)

        # Parameter representing the initial energy of the network
        model.E_0 = pyo.Param(
            initialize=lambda model: self.network.remaining_energy())

        print(f"Initial energy: {model.E_0.value}")

        # Parameter representing the current energy level of all nodes
        # Define the dynamic energy parameter function
        def dynamic_energy_parameter_rule(model, node, round):
            if round == 1:
                return model.E_0
            else:
                energy_consumed = model.energy[node, round - 1]

                # Calculate energy spent by non-cluster head nodes
                non_ch_energy = sum(
                    self.net_model.calculate_energy_tx_non_ch(
                        model.distance[node, other_node]) *
                    model.y[node, other_node, round]
                    for other_node in model.nodes)

                energy_consumed -= non_ch_energy
                return energy_consumed

        # Define the energy parameter as a mutable parameter with the dynamic rule
        model.energy = pyo.Param(
            model.nodes, model.rounds,
            initialize=model.E_0,  # Initial value (can be anything)
            mutable=True
            # within=pyo.NonNegativeReals  # You can specify the domain if needed
        )

        model.energy.update_rule = dynamic_energy_parameter_rule

        # Objective function: Maximize the remaining energy of the network
        def objective_rule(model):
            return sum(model.energy[node, round] for node in model.nodes for round in model.rounds)

        model.objective = pyo.Objective(
            rule=objective_rule, sense=pyo.maximize)

        # Constraint: Ensure that only one round is the first death round
        def unique_first_death_round_rule(model):
            return sum(model.first_death_round[r] for r in model.rounds) == 1

        model.unique_first_death_round_constraint = pyo.Constraint(
            rule=unique_first_death_round_rule)

        # Energy consumption constraint
        def energy_consumption_rule(model, node, round):
            # print(f"Energy consumption rule: {node}, {round}")
            if round == 1:
                return model.energy[node, round] == model.E_0
            else:
                # distance from non-ch to ch
                energy_consumed = model.energy[node, round - 1]

                # Calculate energy spent by non-cluster head nodes
                non_ch_energy = sum(
                    self.net_model.calculate_energy_tx_non_ch(
                        model.distance[node, other_node]) *
                    model.y[node, other_node, round]
                    for other_node in model.nodes)

                # Calculate energy spent by cluster heads
                # ch_energy = self.net_model.calculate_energy_tx_ch(
                #     model.distance[node, 1]) *\
                #     model.cluster_heads[node, round]

                # Calculate energy spent by cluster heads to receive data from non-cluster heads
                # ch_rx_energy = sum(
                #     self.net_model.calculate_energy_rx() *
                #     model.y[non_ch, node, round]
                # for non_ch in model.nodes)

                # Calculate energy spent by control packets
                # pkt_size = 400
                # energy = self.net_model.elect * pkt_size
                # alive_nodes = sum(model.alive[node, round]
                #                 for node in model.nodes)
                # ctrl_energy = energy * alive_nodes

                # Should we subtract the energy spent by control packets? cluster_head_change
                energy_consumed -= non_ch_energy
                # energy_consumed -= ch_energy
                # energy_consumed -= ch_rx_energy
                # energy_consumed -= ctrl_energy
                # energy_consumed += ctrl_energy * \
                #     (1-model.cluster_head_change[round])

            return model.energy[node, round] == energy_consumed

        model.energy_consumption_constraint = pyo.Constraint(
            model.nodes, model.rounds, rule=energy_consumption_rule)

        # Constraint that sets the alive to 0 if the energy of the node at that round is 0
        def alive_rule(model, node, round):
            return model.energy[node, 1] - model.energy[node, round] >= 0

        model.alive_constraint = pyo.Constraint(
            model.nodes, model.rounds, rule=alive_rule)

        # Constraint for first death round
        # def first_death_round_rule(model, round):
        #     expr = 0
        #     for node in model.nodes:
        #         expr += (1-model.alive[node, round])
        #     return model.first_death_round[round] >= expr

        # model.first_death_round_constraint = pyo.Constraint(
        #     model.rounds, rule=first_death_round_rule)

        # Constraint to limit the number of cluster heads per round
        def cluster_heads_limit_rule(model, round):
            return sum(model.cluster_heads[node, round] for node in model.nodes) <= 2

        model.cluster_heads_limit_constraint = pyo.Constraint(
            model.rounds, rule=cluster_heads_limit_rule)

        # Constraint to only select cluster heads from alive nodes
        # def cluster_heads_alive_nodes_rule(model, node, round):
        #     return model.cluster_heads[node, round] <= model.alive[node, round]

        # model.cluster_heads_alive_nodes_constraint = pyo.Constraint(
        #     model.nodes, model.rounds, rule=cluster_heads_alive_nodes_rule)

        # consistency_rule for y
        def consistency_rule(model, i, j, round):
            return model.y[i, j, round] <= model.cluster_heads[j, round]

        model.consistency_constraint = pyo.Constraint(
            model.nodes, model.nodes, model.rounds, rule=consistency_rule)

        # Assign cluster heads to nodes
        def assignment_rule(model, i, round):
            return sum(model.y[i, j, round] for j in model.nodes) == 1

        model.assignment_constraint = pyo.Constraint(
            model.nodes, model.rounds, rule=assignment_rule)

        # Constraint to ensure the cluster head change variable reflects changes in cluster heads
        # def cluster_head_change_rule(model, round):
        #     if round == 1:
        #         return pyo.Constraint.Skip
        #     else:
        #         return model.cluster_head_change[round] >= sum(model.cluster_heads[node, round] - model.cluster_heads[node, round - 1] for node in model.nodes)

        # model.cluster_head_change_constraint = pyo.Constraint(
        #     model.rounds, rule=cluster_head_change_rule)
        # Solve the model
        solver = pyo.SolverFactory("mindtpy")
        results = solver.solve(model)

        if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print(f"Model solved to optimality")
        else:
            print("Model was not solved to optimality.")

        return model

    def run(self):
        print(f"Running {self.name} protocol...")
        num_rounds = self.config.network.protocol.rounds
        plot_clusters_flag = self.config.network.plot
        plot_refresh = self.config.network.plot_refresh

        for node in self.network:
            node.is_cluster_head = False

        # Set all dst_to_sink for all nodes
        for node in self.network:
            node.dst_to_sink = self.network.distance_to_sink(node)

        # data = self.solve_milp()

        # # Save data into a JSON file
        name = "leach_rt_data"

        # try:
        #     os.makedirs("data")
        # except OSError as e:
        #     if e.errno != errno.EEXIST:
        #         raise
        # with open('data/' + name + '.json', 'w') as outfile:
        #     json.dump(data, outfile)

        # load the JSON file
        with open('data/' + name + '.json', 'r') as json_file:
            # Load the JSON data into a Python dictionary
            data = json.load(json_file)

        # Now solve the LIP to find the optimal R
        self.solve_optimal_r(data=data)

        if not plot_clusters_flag:
            self.run_without_plotting(
                num_rounds)
        else:
            self.run_with_plotting(
                num_rounds, plot_refresh)

    def evaluate_round(self, round):

        if (round >= self.start_round + self.num_rounds):
            # print("Generating new set of clusters...")
            self.threshold_energy = D(
                f"{self.network.average_remaining_energy()}")
            self.max_chs = np.ceil(
                self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage)
            # Call a function that generates the new set of clusters
            chs, non_chs = LEACH_CE_E.choose_cluster_heads(
                network=self.network, threshold_energy=self.threshold_energy, max_chs=self.max_chs)
            leach_milp.update_cluster_heads(
                network=self.network, chs=chs)
            leach_milp.update_chs_to_nodes(
                network=self.network, assignments=non_chs)
            # Generate a new set of clusters
            self.start_round = round
            self.num_rounds = self.calculate_r(round=round)
            # print(f"R at round {round}: {self.num_rounds}")
            self.net_model.dissipate_energy(round=round)
            # input(f"Cluster heads at round {round}: {chs}")
            round += 1
            return round

        # We still in the same set of clusters
        self.net_model.dissipate_energy(round=round)
        round += 1
        # input(f"We are still in the same set of clusters at round {round}")

        return round

    def select_cluster_heads(self):
        nodes = [node.node_id for node in self.network]
        # Select self.max_chs cluster heads from the network
        chs = np.random.choice(
            nodes, size=int(self.max_chs), replace=False)
        print(f"Cluster heads: {chs}")
        # Set them as cluster heads
        for node_id in chs:
            self.network.mark_as_cluster_head(
                self.network.get_node(node_id), node_id)
        # Create clusters
        self.network.create_clusters()

    def run_without_plotting(self, num_rounds):
        round = 0
        # Create a random initial network
        self.max_chs = np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage)
        self.select_cluster_heads()
        self.num_rounds = self.calculate_r(round=0)
        self.start_round = 0
        print(f"R: {self.num_rounds}")
        with Progress() as progress:
            task = progress.add_task(
                f"[red]Running {self.name}...", total=num_rounds)
            while self.network.alive_nodes() > 0 and round < num_rounds:
                round = self.evaluate_round(round)
                progress.update(task, completed=round)
            progress.update(task, completed=num_rounds)

    def run_with_plotting(self, num_rounds):
        fig, ax = plt.subplots()
        common.plot_clusters(network=self.network, round=0, ax=ax)

        # Create a random initial network
        self.max_chs = np.ceil(
            self.network.alive_nodes() * self.config.network.protocol.cluster_head_percentage)
        self.select_cluster_heads()
        self.num_rounds = self.calculate_r(round=0)
        self.start_round = 0

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
