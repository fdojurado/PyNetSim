import json
import os
from pynetsim.config.network import NetworkConfig
from pynetsim.leach.hrl.leach_rm import LEACH_RM
from pynetsim.leach.hrl.leach_hrl import LEACH_HRL
from pynetsim.leach.hrl.leach_add import LEACH_ADD
from pynetsim.leach.leach_c import LEACH_C
from pynetsim.leach.leach import LEACH

SELF_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(SELF_PATH, "default.json")

PROTOCOLS = {
    "LEACH": LEACH,
    "LEACH-C": LEACH_C,
    "LEACH-HRL": LEACH_HRL,
    "LEACH-ADD": LEACH_ADD,
    "LEACH-RM": LEACH_RM
}


class PyNetSimConfig:

    def __init__(self, name="default", network=None):
        self.name = name
        self.network = network

    @classmethod
    def from_json(cls, filename=None):

        # If there is not filename, we load the default config
        if filename is None:
            filename = DEFAULT_CONFIG

        try:
            with open(filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {filename} not found.")

        return cls(name=config.get("name", "default"),
                   network=NetworkConfig.from_json(
                       config.get("network")))
