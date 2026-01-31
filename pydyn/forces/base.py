import cupy as cp
from typing import List


class ForceModel:

    implemented_properties: List[str] = []

    def __init__(self):
        self.state = None
        self.results = {}

    def check_state(self, state, tol=1e-15):
        return state == self.state

    def compute(self, state, context):
        pass
        # if state.configure_same_as(self.state):
        #     return
        # self.state = state.copy()
        # self.results = {}

    def build_neighbor_list(self, state, cutoff):
        pos = state.r
        cell = state.box
        pbc = state.pbc
        N = state.N

        # 1. 计算每个方向需要的重复数
        V = cp.abs(cp.linalg.det(cell))
        d1 = V / cp.linalg.norm(cp.cross(cell[1], cell[2]))
        d2 = V / cp.linalg.norm(cp.cross(cell[2], cell[0]))
        d3 = V / cp.linalg.norm(cp.cross(cell[0], cell[1]))
        reps = [
            int(cp.ceil(cutoff / d1)) if pbc[0] else 0,
            int(cp.ceil(cutoff / d2)) if pbc[1] else 0,
            int(cp.ceil(cutoff / d3)) if pbc[2] else 0,
        ]

        # 2. 生成 PBC image 偏移
        shifts = cp.stack(
            cp.meshgrid(
                cp.arange(-reps[0], reps[0] + 1),
                cp.arange(-reps[1], reps[1] + 1),
                cp.arange(-reps[2], reps[2] + 1),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(
            -1, 3
        )  # (Nc,3)
        offsets = shifts @ cell  # Cartesian偏移

        # 3. 构建所有原子对
        i_idx = cp.repeat(cp.arange(N), N * len(offsets))
        j_idx = cp.tile(cp.repeat(cp.arange(N), len(offsets)), N)
        offset_idx = cp.tile(cp.arange(len(offsets)), N * N)

        # 4. 对第二个原子加上 PBC 偏移
        pos_i = pos[i_idx]
        pos_j = pos[j_idx] + offsets[offset_idx]

        # 5. 计算距离平方
        dist2 = cp.sum((pos_i - pos_j) ** 2, axis=1)

        # 6. Mask: cutoff 和自作用
        mask = (dist2 <= cutoff**2) & (dist2 > 1e-8)

        index1 = i_idx[mask]
        index2 = j_idx[mask]
        offset_vec = offsets[offset_idx][mask]

        return index1, index2, offset_vec
