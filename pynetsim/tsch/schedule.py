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
