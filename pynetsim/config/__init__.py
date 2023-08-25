import json
import os
from pynetsim.config.network import NetworkConfig

SELF_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(SELF_PATH, "default.json")


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
