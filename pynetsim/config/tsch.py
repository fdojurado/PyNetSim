

class TSCHConfig:

    def __init__(self, slotframe=None, slot_size=None, channels=None):
        self.slotframe = slotframe
        self.slot_size = slot_size
        self.channels = channels

    @classmethod
    def from_json(cls, json_object=None):
        if json_object is None:
            json_object = {}

        return cls(slotframe=json_object.get("slotframe"),
                   slot_size=json_object.get("slot_size"),
                   channels=json_object.get("channels"))
