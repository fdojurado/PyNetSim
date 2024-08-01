import yaml
import os

from pynetsim.leach.rl.leach_rl import LEACH_RL
from pynetsim.network.simple_model import Simple
from pynetsim.network.extended_model import Extended
from pynetsim.leach.surrogate.surrogate import SurrogateModel
from pynetsim.leach.leach_milp.leach_ce_d import LEACH_CE_D
from pynetsim.leach.leach_milp.leach_ce_e import LEACH_CE_E
from pynetsim.leach.leach_ce import LEACH_CE
from pynetsim.leach.state_of_art.ec_leach import EC_LEACH
from pynetsim.leach.state_of_art.ee_leach import EE_LEACH
from pynetsim.leach.leach_rt import LEACH_RT
from pynetsim.leach.leach_r import LEACH_R
from pynetsim.leach.leach_k import LEACH_K
from pynetsim.leach.leach_c import LEACH_C
from pynetsim.leach.leach import LEACH

SELF_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(SELF_PATH, "default.json")

PROTOCOLS = {
    "LEACH": LEACH,
    "LEACH-C": LEACH_C,
    "LEACH-RL": LEACH_RL,
    "LEACH-K": LEACH_K,
    "LEACH-CE": LEACH_CE,
    "LEACH-CE-D": LEACH_CE_D,
    "LEACH-CE-E": LEACH_CE_E,
    "LEACH-R": LEACH_R,
    "LEACH-RT": LEACH_RT,
    "EC-LEACH": EC_LEACH,
    "EE-LEACH": EE_LEACH,
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


class NodeConfiguration:
    def __init__(self, node_dict):
        self.node_id = node_dict.get('node_id')
        self.x = node_dict.get('x')
        self.y = node_dict.get('y')
        self.type_node = node_dict.get('type_node', DEFAULT_TYPE_NODE)
        self.energy = node_dict.get('energy', DEFAULT_NODE_ENERGY)


class NetworkConfiguration:
    def __init__(self, network_dict):
        self.num_sensor = network_dict.get('num_sensor', DEFAULT_NUM_SENSOR)
        self.transmission_range = network_dict.get(
            'transmission_range', DEFAULT_TRANSMISSION_RANGE)
        self.model = network_dict.get('model', DEFAULT_DEFAULT_MODEL)
        self.plot = network_dict.get('plot', DEFAULT_PLOT)
        self.plot_refresh = network_dict.get('plot_refresh', DEFAULT_PLOT_REFRESH)
        self.protocol = ProtocolConfiguration(
            network_dict.get('protocol', {}))
        self.width = network_dict.get('width', DEFAULT_WIDTH)
        self.height = network_dict.get('height', DEFAULT_HEIGHT)
        self.num_sink = network_dict.get('num_sink', DEFAULT_NUM_SINK)
        self.nodes = [NodeConfiguration(node)
                      for node in network_dict.get('nodes', [])]


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


class Configuration:
    def __init__(self, config_dict):
        self.name = config_dict.get('name')
        self.seed = config_dict.get('seed', DEFAULT_SEED)
        self.surrogate = SurrogateConfiguration(
            config_dict.get('surrogate', {}))
        self.network = NetworkConfiguration(config_dict.get('network', {}))


def load_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as config_file:
            config_data = yaml.safe_load(config_file)
            if config_data:
                return Configuration(config_data)
            raise ValueError("Empty configuration file")
    except FileNotFoundError as ex:
        raise FileNotFoundError(f'Config file not found at {file_path}') from ex
    except yaml.YAMLError as e:
        raise ValueError(f'Error parsing YAML file: {e}') from e
