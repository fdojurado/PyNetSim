import numpy as np
import pynetsim.common as common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pyomo.environ as pyo
import pynetsim.leach.leach_milp as leach_milp


from pynetsim.utils import PyNetSimLogger
from rich.progress import Progress
from pynetsim.leach.surrogate.model import SurrogateModel, NetworkDataset
from torch.utils.data import DataLoader

# -------------------- Create logger --------------------
logger_utility = PyNetSimLogger(log_file="my_log.log", namespace=__name__)
logger = logger_utility.get_logger()


class SURROGATE:

    def __init__(self, network, net_model: object):
        self.name = "SURROGATE"
        self.net_model = net_model
        self.config = network.config
        self.network = network
        self.largest_weight = self.config.surrogate.largest_weight
        self.alpha = 3.3
        self.beta = 0.9
        self.gamma = 1.7
        self.surrogate_model = SurrogateModel(config=self.config)

    def run(self):
        logger.info(f"Running {self.name}...")
        self.model, _, _ = self.surrogate_model.get_model()
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

    def create_input_data(self):
        # We need to get alpha, beta, gamma, and energy levels for all nodes
        # We need to get alpha, beta, gamma for all nodes
        weights = [self.alpha/self.largest_weight, self.beta /
                   self.largest_weight, self.gamma/self.largest_weight]
        w_np = np.array(weights)
        # logger.info(f"Weights: {w_np}, shape: {w_np.shape}")
        # print all energy levels
        # for node in self.network:
        #     logger.info(f"Node {node.node_id} energy level: {node.remaining_energy}")
        # We need to get energy levels for all nodes
        energy_levels = np.array(
            [node.remaining_energy for node in self.network if node.node_id != 1])
        # logger.info(
        #     f"Energy levels: {energy_levels}, shape: {energy_levels.shape}")

        # We need to get the current membership of all nodes, this involves in getting the
        # cluster id of all nodes.
        cluster_ids = np.array(
            [0 if node.cluster_id is None else node.cluster_id for node in self.network])

        # logger.info(f"Cluster ids: {cluster_ids}, shape: {cluster_ids.shape}")
        # Add a zero to the start of the cluster ids
        cluster_ids = np.insert(cluster_ids, 0, 0)

        # logger.info(f"Cluster ids: {cluster_ids}, shape: {cluster_ids.shape}")

        # Now we concatenate weights, and energy levels
        input_data = np.concatenate((w_np, energy_levels))
        # logger.info(f"Input data: {input_data}, shape: {input_data.shape}")

        return input_data, cluster_ids

    def predict_cluster_assignments(self):
        # Before we can predict the cluster assignments, we need to create the input data
        # for the surrogate model.
        numerical_data, categorical_data = self.create_input_data()
        # logger.info(
        #     f"Shapes of numerical and categorical data: {numerical_data.shape}, {categorical_data.shape}")
        # Convert the numerical data to a tensor
        numerical_data = torch.from_numpy(numerical_data.astype(np.float32))
        # Convert the categorical data to a tensor
        categorical_data = torch.from_numpy(categorical_data.astype(np.int64))
        # Unsqueeze the numerical data
        numerical_data = numerical_data.unsqueeze(0)
        # Unsqueeze the categorical data
        categorical_data = categorical_data.unsqueeze(0)
        # print shapes
        # logger.info(
        #     f"Shapes of numerical and categorical data: {numerical_data.shape}, {categorical_data.shape}")
        # Now we can predict the cluster assignments

        self.model.eval()
        # logger.info(
        #     f"Shapes: {numerical_data.shape}, {categorical_data.shape}")
        with torch.no_grad():
            # logger.info(
            #     f"Categorical data: {categorical_data}, shape: {categorical_data.shape}")
            output = self.model(categorical_data=categorical_data,
                           numerical_data=numerical_data)
            # logger.info(f"Output: {output}, shape: {output.shape}")
            _, predicted_assignments = torch.max(output, dim=1)
            # logger.info(
            #     f"Predicted cluster assignments: {predicted_assignments}")
        # Convert the predicted assignments to a numpy array
        predicted_assignments = predicted_assignments.numpy()
        return predicted_assignments[0]

    def generate_initial_cluster_assignments(self):
        # Select num_chs cluster heads from the network
        # chs = np.random.choice(
        #     [node.node_id for node in self.network if node.node_id != 1], size=int(self.num_chs), replace=False)
        # logger.info(f"Cluster heads: {chs}")
        # # convert chs to int
        # chs = [int(ch) for ch in chs]
        # # Set them as cluster heads
        # for node_id in chs:
        #     node = self.network.get_node(node_id)
        #     self.network.mark_as_cluster_head(node, node_id)
        # # Create clusters
        # self.network.create_clusters()
        chs = [2, 14, 98, 21, 58]
        # Mark them as cluster heads
        for cluster_id in chs:
            node = self.network.get_node(cluster_id)
            self.network.mark_as_cluster_head(node, cluster_id)
        cluster_assignments = {
            "2": 2, "3": 14, "4": 98, "5": 21, "6": 2, "7": 21, "8": 14, "9": 14, "10": 98, "11": 21, "12": 98, "13": 21, "14": 14, "15": 2, "16": 14, "17": 14, "18": 14, "19": 21, "20": 21, "21": 21, "22": 2, "23": 58, "24": 2, "25": 2, "26": 21, "27": 21, "28": 58, "29": 2, "30": 58, "31": 14, "32": 21, "33": 14, "34": 21, "35": 14, "36": 98, "37": 14, "38": 21, "39": 14, "40": 2, "41": 58, "42": 14, "43": 58, "44": 14, "45": 14, "46": 21, "47": 21, "48": 14, "49": 21, "50": 58, "51": 98, "52": 98, "53": 2, "54": 14, "55": 21, "56": 58, "57": 21, "58": 58, "59": 2, "60": 2, "61": 14, "62": 21, "63": 21, "64": 14, "65": 21, "66": 2, "67": 98, "68": 98, "69": 14, "70": 98, "71": 14, "72": 14, "73": 21, "74": 21, "75": 14, "76": 14, "77": 2, "78": 2, "79": 21, "80": 21, "81": 21, "82": 2, "83": 14, "84": 14, "85": 21, "86": 58, "87": 98, "88": 2, "89": 21, "90": 58, "91": 98, "92": 21, "93": 98, "94": 21, "95": 14, "96": 21, "97": 14, "98": 98, "99": 98, "100": 21
        }
        for node in self.network:
            if node.node_id not in chs and node.node_id != 1:
                self.network.mark_as_non_cluster_head(node)
                cluster_head_id = cluster_assignments[str(node.node_id)]
                cluster_head = self.network.get_node(cluster_head_id)
                node.cluster_id = cluster_head_id
                node.dst_to_cluster_head = self.network.distance_between_nodes(
                    node, cluster_head)

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

    def choose_cluster_heads(self, cluster_assignments):
        alive_nodes = [node.node_id for node in self.network if not self.network.should_skip_node(
            node) and self.network.alive(node)]

        # Check if cluster assignments is empty
        if len(cluster_assignments) == 0:
            # Potential cluster heads
            cluster_ids = [node for node in alive_nodes if self.network.get_node(
                node).remaining_energy >= self.threshold_energy]
        else:
            cluster_ids = np.unique(cluster_assignments)
            # Remove node id 0
            cluster_ids = cluster_ids[cluster_ids != 0]
            # Convert cluster ids to int
            cluster_ids = [int(cluster_id) for cluster_id in cluster_ids]

        model = self.create_model(cluster_ids, alive_nodes)

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

    def create_clusters(self, cluster_assignments):
        # Get the unique cluster ids, they are the cluster heads
        cluster_ids = np.unique(cluster_assignments)
        # Remove node id 0
        cluster_ids = cluster_ids[cluster_ids != 0]
        # Convert cluster ids to int
        cluster_ids = [int(cluster_id) for cluster_id in cluster_ids]
        logger.info(f"Predicted cluster ids: {cluster_ids}")
        # Check which cluster heads has the energy above the threshold
        energy_th = self.network.average_remaining_energy()
        logger.info(f"Energy threshold: {energy_th}")
        # Get the cluster heads that has the energy above the threshold
        cluster_ids = [
            cluster_id for cluster_id in cluster_ids if self.network.get_node(cluster_id).remaining_energy >= energy_th]
        logger.info(f"Cluster ids above threshold: {cluster_ids}")
        # Sort the cluster ids by their remaining energy
        cluster_ids = sorted(
            cluster_ids, key=lambda cluster_id: self.network.get_node(cluster_id).remaining_energy, reverse=True)
        logger.info(f"Sorted cluster ids: {cluster_ids}")
        # print all nodes that have the energy above the threshold
        nodes_above_threshold = []
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            if node.remaining_energy >= energy_th:
                nodes_above_threshold.append(node.node_id)
        logger.info(f"Nodes above threshold: {nodes_above_threshold}")
        # Mark them as cluster heads
        for cluster_id in cluster_ids:
            node = self.network.get_node(cluster_id)
            self.network.mark_as_cluster_head(node, cluster_id)
        # Create clusters
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            cluster_id = int(cluster_assignments[node.node_id])
            if cluster_id != 0:
                ch = self.network.get_node(cluster_id)
                node.dst_to_cluster_head = self.network.distance_between_nodes(
                    node, ch)
                node.cluster_id = cluster_id
            else:
                # Get the closest cluster head
                distances = {
                    ch.node_id: self.network.distance_between_nodes(node, ch)
                    for ch in self.network if ch.is_cluster_head
                }
                cluster_id = min(distances, key=distances.get)
                min_distance = distances[cluster_id]
                node.dst_to_cluster_head = min_distance
                node.cluster_id = cluster_id

    def evaluate_round(self, round):
        round += 1

        self.max_chs = int(self.network.alive_nodes() *
                           self.config.network.protocol.cluster_head_percentage) + 1

        if round == 1:
            for node in self.network:
                self.network.mark_as_non_cluster_head(node)

            # We need to get the initial cluster assignments
            self.generate_initial_cluster_assignments()
        else:
            self.threshold_energy = self.network.average_remaining_energy()
            # Create cluster assignments predicted by the surrogate model
            cluster_assignments = self.predict_cluster_assignments()
            for node in self.network:
                self.network.mark_as_non_cluster_head(node)
            # create cluster
            chs, node_cluster_head = self.choose_cluster_heads(
                cluster_assignments)
            leach_milp.update_cluster_heads(self.network, chs)
            leach_milp.update_chs_to_nodes(self.network, node_cluster_head)
            # print(f"Cluster heads at round {round}: {chs}")
            # print(f"Cluster assignments at round {round}: {node_cluster_head}")
        self.net_model.dissipate_energy(round=round)
        # input("Press enter to continue...")
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

    def run_with_plotting(self, num_rounds):
        fig, ax = plt.subplots()
        common.plot_clusters(network=self.network, round=0, ax=ax)

        def animate(round):
            round = self.evaluate_round(round)

            if round >= num_rounds or self.network.alive_nodes() <= 0:
                logger.info("Done!")
                ani.event_source.stop()

            ax.clear()
            common.plot_clusters(network=self.network, round=round, ax=ax)

            plt.pause(0.1)

        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, num_rounds + 1), repeat=False)

        plt.show()
