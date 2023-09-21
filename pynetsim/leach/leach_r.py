import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyomo.environ as pyo
import pynetsim.leach.leach_milp as leach_milp


from pynetsim.leach.leach_milp.leach_ce_e import LEACH_CE_E
from rich.progress import Progress
from decimal import Decimal as D


class LEACH_R:

    def __init__(self, network, net_model: object):
        self.name = "LEACH-R"
        self.net_model = net_model
        self.elect = self.net_model.elect
        self.config = network.config
        self.network = network

    def calculate_expected_energy_consumption_per_round(self):
        # Calculate the energy spent by non-cluster heads
        E_non_ch = 0
        for node in self.network:
            if self.network.should_skip_node(node) or node.is_cluster_head:
                continue
            # Get the cluster head of the node
            cluster_head = self.network.get_cluster_head(node)
            # Calculate the distance between the node and its cluster head
            distance = self.network.distance_between_nodes(node, cluster_head)
            # Calculate the energy spent by the node to transmit to its cluster head
            E_non_ch += self.net_model.calculate_energy_tx_non_ch(distance)
        # Calculate the energy spent by cluster heads
        E_ch = 0
        E_ch_rx = 0
        for node in self.network:
            if self.network.should_skip_node(node) or not node.is_cluster_head:
                continue
            # Calculate the energy spent by the node to transmit to the sink
            E_ch += self.net_model.calculate_energy_tx_ch(node.dst_to_sink)
            # Calculate the energy spent by the node to receive data from non-cluster heads
            E_ch_rx += self.net_model.calculate_energy_rx()
        # Calculate the energy spent by control packets
        E_control = 0
        # Get the number of cluster heads
        chs = self.network.num_cluster_heads()
        pkt_size = (4*chs+15) * 8
        energy = self.elect * pkt_size
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            E_control += energy

        return E_non_ch + E_ch + E_ch_rx + E_control, E_control

    def calculate_expected_energy_consumption(self, num_rounds: int):
        E, E_control = self.calculate_expected_energy_consumption_per_round()
        E_R = num_rounds*E - (num_rounds-1)*E_control
        return E_R

    def objective_function(self, model):
        print("Calculating objective function...")
        # Calculate expected energy consumption over R rounds
        E_R = self.calculate_expected_energy_consumption(num_rounds=model.R)
        # Calculate the network lifetime
        NL = model.init_energy / (model.init_energy-E_R)

        return -NL

    def create_model(self):
        model = pyo.ConcreteModel()
        # Decision variables
        model.R = pyo.Var(domain=pyo.NonNegativeIntegers)

        model.init_energy = pyo.Param(
            initialize=self.network.remaining_energy())

        # Objective function
        model.obj = pyo.Objective(
            sense=pyo.minimize, rule=self.objective_function)
        # print objective expression
        # print(f"Objective function: {model.obj.expr}")

        # Constraints

        model.R_constraint = pyo.Constraint(
            expr=model.R >= 1)

        model.nl_denominator_constraint = pyo.Constraint(
            expr=model.init_energy - self.calculate_expected_energy_consumption(num_rounds=model.R) >= 0)

        # Total energy consumption constraint
        def total_energy_consumption_constraint(model):
            total_energy_consumption = self.calculate_expected_energy_consumption(
                num_rounds=model.R)
            return total_energy_consumption <= model.init_energy * 1/100

        model.total_energy_consumption_constraint = pyo.Constraint(
            rule=total_energy_consumption_constraint)

        return model

    def calculate_r(self, round):
        print(f"Calculating R for round {round}...")

        model = self.create_model()

        # Solve the model
        solver = pyo.SolverFactory('mindtpy')
        results = solver.solve(model)

        if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print(f"Model solved to optimality: {model.R.value}")
            return pyo.value(model.R)
        else:
            print("Model was not solved to optimality.")
            print("Solver Status: ", results.solver.status)
            print("Termination Condition: ",
                  results.solver.termination_condition)
            return 1

    def run(self):
        print(f"Running {self.name} protocol...")
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

            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
