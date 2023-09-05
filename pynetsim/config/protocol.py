DEFAULT_DISTANCE = 75


class ProtocolConfig():
    """Protocol configuration class."""

    def __init__(self, name=None, init_energy=None,
                 rounds=None, cluster_head_percentage=None, dst_threshold=None,
                 eelect_nano=None, eamp_pico=None, efs_pico=None,
                 eda_nano=None, packet_size=None, max_steps=None):
        self.name = name
        self.init_energy = init_energy
        self.rounds = rounds
        self.cluster_head_percentage = cluster_head_percentage
        self.dst_threshold = dst_threshold
        self.eelect_nano = eelect_nano
        self.eamp_pico = eamp_pico
        self.efs_pico = efs_pico
        self.eda_nano = eda_nano
        self.packet_size = packet_size
        self.max_steps = max_steps

    @classmethod
    def from_json(cls, json_object=None):
        if json_object is None:
            json_object = {}

        return cls(name=json_object.get("name", None),
                   init_energy=json_object.get("init_energy", None),
                   rounds=json_object.get("rounds", None),
                   cluster_head_percentage=json_object.get(
            "cluster_head_percentage", None),
            dst_threshold=json_object.get("dst_threshold", DEFAULT_DISTANCE),
            eelect_nano=json_object.get("eelect_nano", None),
            eamp_pico=json_object.get("eamp_pico", None),
            efs_pico=json_object.get("efs_pico", None),
            eda_nano=json_object.get("eda_nano", None),
            packet_size=json_object.get("packet_size", None),
            max_steps=json_object.get("max_steps", None))
