import yaml
import os

from pynetsim.leach.rl.leach_rl import LEACH_RL
from pynetsim.leach.rl.leach_rl_mult import LEACH_RL_MULT
from pynetsim.leach.rl.leach_rl_loss import LEACH_RL_LOSS
from pynetsim.leach.rl.leach_hrl import LEACH_HRL
from pynetsim.network.simple_model import Simple
from pynetsim.network.extended_model import Extended
from pynetsim.leach.surrogate.surrogate import SURROGATE
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
    "LEACH-RL-LOSS": LEACH_RL_LOSS,
    "LEACH-HRL": LEACH_HRL,
    "LEACH-RL-MULT": LEACH_RL_MULT,
    "LEACH-K": LEACH_K,
    "LEACH-CE": LEACH_CE,
    "LEACH-CE-D": LEACH_CE_D,
    "LEACH-CE-E": LEACH_CE_E,
    "LEACH-R": LEACH_R,
    "LEACH-RT": LEACH_RT,
    "EC-LEACH": EC_LEACH,
    "EE-LEACH": EE_LEACH,
    "Surrogate": SURROGATE
}

NETWORK_MODELS = {
    "simple": Simple,
    "extended": Extended
}


DEFAULT_NUM_SENSOR = 100
DEFAULT_TRANSMISSION_RANGE = 80
DEFAULT_WIDTH = 200
DEFAULT_HEIGHT = 200
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
        self.protocol = ProtocolConfiguration(
            network_dict.get('protocol', {}))
        self.width = network_dict.get('width', DEFAULT_WIDTH)
        self.height = network_dict.get('height', DEFAULT_HEIGHT)
        self.num_sink = network_dict.get('num_sink', DEFAULT_NUM_SINK)
        self.nodes = [NodeConfiguration(node)
                      for node in network_dict.get('nodes', [])]


# Surrogate defaults
HIDDEN_DIM = 512
LSTM_HIDDEN = 512
OUTPUT_DIM = 101
LEARNING_RATE = 1e-6
EPOCHS = 5000
LARGEST_WEIGHT = 6
LARGEST_ENERGY_LEVEL = 4
NUM_CLUSTERS = 100
NUM_EMBEDDINGS = 101
EMBEDDING_DIM = 30
NUMERICAL_DIM = 102
WEIGHT_DECAY = 1e-5
DROP_OUT = 0.2

# Data loader
BATCH_SIZE = 64
TEST_RATIO = 0.2
NUM_WORKERS = 1

# Print and plot intervals
PRINT_EVERY = 1
PLOT_EVERY = 10
EVAL_EVERY = 10


class SurrogateConfiguration:
    def __init__(self, surrogate_dict):
        # Surrogate model
        self.lstm_arch = surrogate_dict.get('lstm_arch', "simple")
        self.epochs = surrogate_dict.get('epochs', EPOCHS)
        self.hidden_dim = surrogate_dict.get(
            'hidden_dim', HIDDEN_DIM)
        self.lstm_hidden = surrogate_dict.get(
            'lstm_hidden', LSTM_HIDDEN)
        self.output_dim = surrogate_dict.get('output_dim', OUTPUT_DIM)
        self.num_clusters = surrogate_dict.get(
            'num_clusters', NUM_CLUSTERS)
        self.num_embeddings = surrogate_dict.get(
            'num_embeddings', NUM_EMBEDDINGS)
        self.embedding_dim = surrogate_dict.get(
            'embedding_dim', EMBEDDING_DIM)
        self.numerical_dim = surrogate_dict.get(
            'numerical_dim', NUMERICAL_DIM)
        self.weight_decay = surrogate_dict.get(
            'weight_decay', WEIGHT_DECAY)
        self.drop_out = surrogate_dict.get('drop_out', DROP_OUT)
        self.batch_size = surrogate_dict.get('batch_size', BATCH_SIZE)
        self.learning_rate = surrogate_dict.get(
            'learning_rate', LEARNING_RATE)
        self.test_ratio = surrogate_dict.get('test_ratio', TEST_RATIO)
        self.largest_weight = surrogate_dict.get(
            'largest_weight', LARGEST_WEIGHT)
        self.largest_energy_level = surrogate_dict.get(
            'largest_energy_level', LARGEST_ENERGY_LEVEL)
        self.num_workers = surrogate_dict.get('num_workers', NUM_WORKERS)
        self.load_model = surrogate_dict.get('load_model', False)
        self.model_path = surrogate_dict.get('model_path', None)
        self.generate_data = surrogate_dict.get('generate_data', False)
        self.raw_data_folder = surrogate_dict.get('raw_data_folder', None)
        self.data_folder = surrogate_dict.get('data_folder', None)
        self.plots_folder = surrogate_dict.get('plots_folder', None)
        # Print and plot intervals
        self.print_every = surrogate_dict.get('print_every', PRINT_EVERY)
        self.plot_every = surrogate_dict.get('plot_every', PLOT_EVERY)
        self.eval_every = surrogate_dict.get('eval_every', EVAL_EVERY)


class Configuration:
    def __init__(self, config_dict):
        self.name = config_dict.get('name')
        self.surrogate = SurrogateConfiguration(
            config_dict.get('surrogate', {}))
        self.network = NetworkConfiguration(config_dict.get('network', {}))


def load_config(file_path):
    try:
        with open(file_path, 'r') as config_file:
            config_data = yaml.safe_load(config_file)
            if config_data:
                return Configuration(config_data)
            else:
                raise ValueError("Empty configuration file")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
