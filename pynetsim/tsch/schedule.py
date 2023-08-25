import types

cell_type = types.SimpleNamespace()
cell_type.UC_RX = 2
cell_type.UC_TX = 1


class TSCHSchedule():
    def __init__(
        self,
        cell_type=None,
        dst_id=None,
        ch=None,
        ts=None
    ) -> None:
        assert isinstance(cell_type, int)
        assert isinstance(dst_id, int | type(None))
        assert isinstance(ch, int)
        assert isinstance(ts, int)
        self.cell_type = cell_type
        self.dst_id = dst_id
        self.ch = ch
        self.ts = ts
