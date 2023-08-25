from pynetsim.config.nodes import NodesConfig
from pynetsim.config.tsch import TSCHConfig

NUM_SENSOR = 20
TRANSMISSION_RANGE = 80
WIDTH = 200
HEIGHT = 200
NUM_SINK = 1


class NetworkConfig:

    def __init__(self, num_sensor=None, nodes=None,
                 transmission_range=None,
                 width=None, height=None,
                 num_sink=None,
                 tsch=None):

        self.num_sensor = num_sensor
        self.transmission_range = transmission_range
        self.width = width
        self.height = height
        self.num_sink = num_sink
        self.nodes = nodes
        self.tsch = tsch

    @classmethod
    def from_json(cls, json_object=None):

        if json_object is None:
            json_object = {}

        return cls(num_sensor=json_object.get("num_sensor", NUM_SENSOR),
                   nodes=NodesConfig.from_json(
            json_object.get("nodes")),
            transmission_range=json_object.get(
                       "transmission_range", TRANSMISSION_RANGE),
            width=json_object.get("width", WIDTH),
            height=json_object.get("height", HEIGHT),
            num_sink=json_object.get("num_sink", NUM_SINK),
            tsch=TSCHConfig.from_json(json_object.get("tsch")))
