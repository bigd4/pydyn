from .libneighbor import NeighborList
import cupy as cp


class CudaNeighborList:
    def __init__(self, N, MN, rc):
        self.nb = NeighborList(N, MN, rc)
        self.idx_i = cp.zeros(N * MN, dtype=cp.int32)
        self.idx_j = cp.zeros(N * MN, dtype=cp.int32)
        self.offset = cp.zeros(N * MN * 3, dtype=cp.float32)
        self.n_neigh = 0

    def find_neighbor(self, state):
        h = state.box.transpose(1, 0).reshape(-1).astype(cp.float32).tolist()
        self.nb.update_box(h, state.pbc[0], state.pbc[1], state.pbc[2])
        x = state.r[:, 0].astype(cp.float32)
        y = state.r[:, 1].astype(cp.float32)
        z = state.r[:, 2].astype(cp.float32)
        n_neigh = self.nb.find_neighbor(
            x.data.ptr,
            y.data.ptr,
            z.data.ptr,
        )
        self.nb.convert_ijS(
            self.idx_i.data.ptr, self.idx_j.data.ptr, self.offset.data.ptr
        )
        return (
            self.idx_i[:n_neigh],
            self.idx_j[:n_neigh],
            self.offset[: 3 * n_neigh].reshape(-1, 3),
        )
