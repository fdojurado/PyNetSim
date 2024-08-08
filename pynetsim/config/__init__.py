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


import yaml
import os
import logging

from pynetsim.leach.rl.leach_rl import LEACH_RL
from pynetsim.network.simple_model import Simple
from pynetsim.network.extended_model import Extended
from pynetsim.leach.surrogate.surrogate import SurrogateModel
from pynetsim.leach.leach_milp.leach_ce_e import LEACH_CE_E
from pynetsim.leach.leach_ee import LEACH_EE
from pynetsim.leach.leach_d import LEACH_D
from pynetsim.leach.leach_cm import LEACH_CM
from pynetsim.leach.leach_c import LEACH_C
from pynetsim.leach.leach import LEACH
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("Main")


SELF_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(SELF_PATH, "default.json")

PROTOCOLS = {
    "LEACH": LEACH,
    "LEACH-C": LEACH_C,
    "LEACH-RL": LEACH_RL,
    "LEACH-CE-E": LEACH_CE_E,
    "LEACH-EE": LEACH_EE,
    "LEACH-CM": LEACH_CM,
    "LEACH-D": LEACH_D,
    "Surrogate": SurrogateModel
}

NETWORK_MODELS = {
    "simple": Simple,
    "extended": Extended
}


DEFAULT_SEED = 42
DEFAULT_NUM_SENSOR = 100
DEFAULT_TRANSMISSION_RANGE = 80
DEFAULT_WIDTH = 200
DEFAULT_HEIGHT = 200
DEFAULT_PLOT = False
DEFAULT_PLOT_REFRESH = 0.1
DEFAULT_NUM_SINK = 1
DEFAULT_DEFAULT_MODEL = "simple"
# Protocol defaults
DEFAULT_PROTOCOL_NAME = "LEACH"
# RL max steps
DEFAULT_MAX_STEPS = 2000
DEFAULT_INIT_ENERGY = 0.5
DEFAULT_ROUNDS = 8000
DEFAULT_CLUSTER_HEAD_PERCENTAGE = 0.05
DEFAULT_EELECT = 50 * 10**(-9)
DEFAULT_EAMP = 0.0013 * 10**(-12)
DEFAULT_EFS = 10 * 10**(-12)
DEFAULT_EDA = 5 * 10**(-9)
DEFAULT_PACKET_SIZE = 4000
# Node defaults
DEFAULT_TYPE_NODE = "Sensor"
DEFAULT_NODE_ENERGY = 0.5


class ProtocolConfiguration:
    def __init__(self, protocol_dict):
        self.name = protocol_dict.get('name', DEFAULT_PROTOCOL_NAME)
        self.max_steps = protocol_dict.get('max_steps', DEFAULT_MAX_STEPS)
        self.init_energy = protocol_dict.get(
            'init_energy', DEFAULT_INIT_ENERGY)
        self.rounds = protocol_dict.get('rounds', DEFAULT_ROUNDS)
        self.cluster_head_percentage = protocol_dict.get(
            'cluster_head_percentage', DEFAULT_CLUSTER_HEAD_PERCENTAGE)
        self.eelect = protocol_dict.get('eelect', DEFAULT_EELECT)
        self.eamp = protocol_dict.get('eamp', DEFAULT_EAMP)
        self.efs = protocol_dict.get('efs', DEFAULT_EFS)
        self.eda = protocol_dict.get('eda', DEFAULT_EDA)
        self.packet_size = protocol_dict.get(
            'packet_size', DEFAULT_PACKET_SIZE)

    def __str__(self):
        name = f"Name: {self.name}\n" if self.name else ""
        max_steps = f"Max steps: {self.max_steps}\n" if self.max_steps else ""
        init_energy = f"Initial energy: {self.init_energy}\n" if self.init_energy else ""
        rounds = f"Rounds: {self.rounds}\n" if self.rounds else ""
        cluster_head_percentage = f"Cluster head percentage: {self.cluster_head_percentage}\n" if self.cluster_head_percentage else ""
        eelect = f"EELECT: {self.eelect}\n" if self.eelect else ""
        eamp = f"EAMP: {self.eamp}\n" if self.eamp else ""
        efs = f"EFS: {self.efs}\n" if self.efs else ""
        eda = f"EDA: {self.eda}\n" if self.eda else ""
        packet_size = f"Packet size: {self.packet_size}\n" if self.packet_size else ""
        return f"{name}{max_steps}{init_energy}{rounds}{cluster_head_percentage}{eelect}{eamp}{efs}{eda}{packet_size}"


class NodeConfiguration:
    def __init__(self, node_dict):
        self.node_id = node_dict.get('node_id')
        self.x = node_dict.get('x')
        self.y = node_dict.get('y')
        self.type_node = node_dict.get('type_node', DEFAULT_TYPE_NODE)
        self.energy = node_dict.get('energy', DEFAULT_NODE_ENERGY)

    def __str__(self):
        node_id = f"Node ID: {self.node_id}\n" if self.node_id else ""
        x = f"X: {self.x}\n" if self.x else ""
        y = f"Y: {self.y}\n" if self.y else ""
        type_node = f"Type: {self.type_node}\n" if self.type_node else ""
        energy = f"Energy: {self.energy}\n" if self.energy else ""
        return f"{node_id}{x}{y}{type_node}{energy}"


class NetworkConfiguration:
    def __init__(self, network_dict):
        self.num_sensor = network_dict.get('num_sensor', DEFAULT_NUM_SENSOR)
        self.transmission_range = network_dict.get(
            'transmission_range', DEFAULT_TRANSMISSION_RANGE)
        self.model = network_dict.get('model', DEFAULT_DEFAULT_MODEL)
        self.plot = network_dict.get('plot', DEFAULT_PLOT)
        self.plot_refresh = network_dict.get(
            'plot_refresh', DEFAULT_PLOT_REFRESH)
        self.protocol = ProtocolConfiguration(
            network_dict.get('protocol', {}))
        self.width = network_dict.get('width', DEFAULT_WIDTH)
        self.height = network_dict.get('height', DEFAULT_HEIGHT)
        self.num_sink = network_dict.get('num_sink', DEFAULT_NUM_SINK)
        self.nodes = [NodeConfiguration(node)
                      for node in network_dict.get('nodes', [])]

    def __str__(self):
        num_sensor = f"Number of sensors: {self.num_sensor}\n" if self.num_sensor else ""
        transmission_range = f"Transmission range: {self.transmission_range}\n" if self.transmission_range else ""
        model = f"Model: {self.model}\n" if self.model else ""
        plot = f"Plot: {self.plot}\n" if self.plot else ""
        plot_refresh = f"Plot refresh: {self.plot_refresh}\n" if self.plot_refresh else ""
        protocol = f"Protocol: \n{self.protocol}\n" if self.protocol else ""
        width = f"Width: {self.width}\n" if self.width else ""
        height = f"Height: {self.height}\n" if self.height else ""
        num_sink = f"Number of sinks: {self.num_sink}\n" if self.num_sink else ""
        nodes = f"Nodes: \n"
        nodes += "\n".join([f"{node}" for node in self.nodes])
        return f"{num_sensor}{transmission_range}{model}{plot}{plot_refresh}{protocol}{width}{height}{num_sink}{nodes}"


# Default surrogate configuration
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.5


class SurrogateConfiguration:
    def __init__(self, surrogate_dict):
        self.cluster_head_model = surrogate_dict.get(
            'cluster_head_model', None)
        self.cluster_head_data = surrogate_dict.get(
            'cluster_head_data', None)
        self.cluster_assignment_model = surrogate_dict.get(
            'cluster_assignment_model', None)
        self.cluster_assignment_data = surrogate_dict.get(
            'cluster_assignment_data', None)
        self.alpha = surrogate_dict.get('alpha', ALPHA)
        self.beta = surrogate_dict.get('beta', BETA)
        self.gamma = surrogate_dict.get('gamma', GAMMA)

    def __str__(self):
        cluster_head_model = f"\tCluster Head Model: {self.cluster_head_model}\n" if self.cluster_head_model else ""
        cluster_head_data = f"\tCluster Head Data: {self.cluster_head_data}\n" if self.cluster_head_data else ""
        cluster_assignment_model = f"\tCluster Assignment Model: {self.cluster_assignment_model}\n" if self.cluster_assignment_model else ""
        cluster_assignment_data = f"\tCluster Assignment Data: {self.cluster_assignment_data}\n" if self.cluster_assignment_data else ""
        alpha = f"\tAlpha: {self.alpha}\n" if self.alpha else ""
        beta = f"\tBeta: {self.beta}\n" if self.beta else ""
        gamma = f"\tGamma: {self.gamma}\n" if self.gamma else ""
        return f"{cluster_head_model}{cluster_head_data}{cluster_assignment_model}{cluster_assignment_data}{alpha}{beta}{gamma}"


class Configuration:
    def __init__(self, config_dict):
        self.name = config_dict.get('name')
        self.seed = config_dict.get('seed', DEFAULT_SEED)
        self.surrogate = SurrogateConfiguration(
            config_dict.get('surrogate', {}))
        self.network = NetworkConfiguration(config_dict.get('network', {}))

    def __str__(self):
        name = f"Name: {self.name}\n" if self.name else ""
        seed = f"Seed: {self.seed}\n" if self.seed else ""
        surrogate = f"Surrogate: \n{self.surrogate}\n" if self.surrogate else ""
        network = f"Network: \n{self.network}\n" if self.network else ""
        return f"{name}{seed}{surrogate}{network}"


def print_rich_table_config(config, network):
    table = Table(title="Configuration")
    table.add_column("Name", style="cyan")
    table.add_column("Seed", style="magenta")
    table.add_row(config.name, str(config.seed))
    console = Console()
    console.print(table)
    # Now lets print the surrogate configuration
    # only print the surrogate configuration if it is not empty
    if config.surrogate.cluster_head_model and config.surrogate.cluster_assignment_model:
        table = Table(title="Surrogate Configuration")
        table.add_column("Cluster Head Model", style="cyan")
        table.add_column("Cluster Head Data", style="magenta")
        table.add_column("Cluster Assignment Model", style="cyan")
        table.add_column("Cluster Assignment Data", style="magenta")
        table.add_column("Alpha", style="cyan")
        table.add_column("Beta", style="magenta")
        table.add_column("Gamma", style="cyan")
        table.add_row(config.surrogate.cluster_head_model, config.surrogate.cluster_head_data,
                      config.surrogate.cluster_assignment_model, config.surrogate.cluster_assignment_data,
                      str(config.surrogate.alpha), str(config.surrogate.beta), str(config.surrogate.gamma))
        console.print(table)
    else:
        logger.warning("Surrogate configuration is empty")
    # Now lets print the network configuration
    table = Table(title="Network Configuration")
    table.add_column("# of sensors", style="cyan")
    table.add_column("Tx range", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Plot", style="magenta")
    table.add_column("Plot refresh", style="cyan")
    table.add_column("Protocol", style="magenta")
    table.add_column("Width", style="cyan")
    table.add_column("Height", style="magenta")
    table.add_column("Number of sinks", style="cyan")
    table.add_row(str(config.network.num_sensor), str(config.network.transmission_range),
                  config.network.model, str(config.network.plot), str(
                      config.network.plot_refresh),
                  str(config.network.protocol.name), str(
                      config.network.width), str(config.network.height),
                  str(config.network.num_sink))
    console.print(table)
    # Now lets print the protocol configuration
    table = Table(title="Protocol Configuration")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Max steps", style="cyan")
    table.add_column("Initial energy", style="magenta")
    table.add_column("Rounds", style="cyan")
    table.add_column("Cluster head %", style="magenta")
    table.add_column("EELECT", style="cyan")
    table.add_column("EAMP", style="magenta")
    table.add_column("EFS", style="cyan")
    table.add_column("EDA", style="magenta")
    table.add_column("Packet size", style="cyan")
    is_centralized = "Centralized" if network.is_centralized() else "Distributed"
    table.add_row(config.network.protocol.name,
                  is_centralized,
                  str(config.network.protocol.max_steps),
                  str(config.network.protocol.init_energy), str(
                      config.network.protocol.rounds),
                  str(config.network.protocol.cluster_head_percentage), str(
                      config.network.protocol.eelect),
                  str(config.network.protocol.eamp), str(
                      config.network.protocol.efs),
                  str(config.network.protocol.eda), str(config.network.protocol.packet_size))
    console.print(table)
    # Now lets print the nodes configuration
    table = Table(title="Nodes Configuration")
    table.add_column("Node ID", style="cyan")
    table.add_column("X", style="magenta")
    table.add_column("Y", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Energy", style="cyan")
    for node in config.network.nodes:
        table.add_row(str(node.node_id), str(node.x), str(node.y),
                      node.type_node, str(node.energy))
    console.print(table)


def load_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as config_file:
            config_data = yaml.safe_load(config_file)
            if config_data:
                return Configuration(config_data)
            raise ValueError("Empty configuration file")
    except FileNotFoundError as ex:
        raise FileNotFoundError(
            f'Config file not found at {file_path}') from ex
    except yaml.YAMLError as e:
        raise ValueError(f'Error parsing YAML file: {e}') from e
