"""Neighbor list for small cells where cell edge < 2 * cutoff.
小胞近邻列表，适用于晶胞边长小于两倍截断半径的情况。

For small periodic cells, a single atom can appear as its own neighbor
through periodic images. This implementation explicitly enumerates all
image cells within cutoff range, making it correct for any cell size.
/ 对于小周期胞，一个原子可以通过周期镜像成为自身的近邻。
此实现显式枚举截断范围内的所有镜像胞，对任意胞大小均正确。

The algorithm:
1. Determine the number of image replicas needed in each lattice direction
   so that all atoms within cutoff are covered.
2. For each image (n1, n2, n3), compute r_j + n·cell - r_i for all (i, j).
3. Keep pairs with distance < cutoff (excluding self-image at origin with i==j).
"""

import cupy as cp
import math


class SmallCellNeighborList:
    """Neighbor list supporting cells of any size with periodic boundaries.
    支持任意大小周期胞的近邻列表。

    Usage:
        nb = SmallCellNeighborList(cutoff=3.0)
        idx_i, idx_j, offsets = nb.find_neighbor(state)

    Args:
        cutoff: Neighbor search cutoff in Angstroms. / 近邻搜索截断半径（埃）。
    """

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def find_neighbor(self, state):
        """Find all neighbor pairs within cutoff under PBC.
        在周期边界条件下查找截断半径内的所有近邻对。

        Args:
            state: Simulation state with r, box, pbc attributes.
                / 具有 r, box, pbc 属性的模拟状态。

        Returns:
            (idx_i, idx_j, offsets):
                idx_i: Atom index i for each pair. / 每对的原子索引 i。
                idx_j: Atom index j for each pair. / 每对的原子索引 j。
                offsets: Displacement vectors r_j - r_i + image, shape (n_pairs, 3).
                    / 位移矢量 r_j - r_i + 镜像，形状 (n_pairs, 3)。
        """
        r = state.r           # (N, 3)
        box = state.box       # (3, 3) — row vectors
        pbc = state.pbc       # (3,) bool
        N = state.N
        rc = self.cutoff

        # --- Determine how many image replicas needed per direction ---
        # For lattice vector a_k, the perpendicular height h_k = V / |a_i × a_j|.
        # We need n_k = ceil(cutoff / h_k) replicas in direction k.
        box_np = cp.asnumpy(box)
        a0, a1, a2 = box_np[0], box_np[1], box_np[2]
        vol = abs(float(cp.linalg.det(box)))

        cross01 = cp.asnumpy(cp.cross(cp.asarray(a0), cp.asarray(a1)))
        cross12 = cp.asnumpy(cp.cross(cp.asarray(a1), cp.asarray(a2)))
        cross20 = cp.asnumpy(cp.cross(cp.asarray(a2), cp.asarray(a0)))

        import numpy as np
        h0 = vol / np.linalg.norm(cross12)
        h1 = vol / np.linalg.norm(cross20)
        h2 = vol / np.linalg.norm(cross01)

        n0 = math.ceil(rc / h0) if pbc[0] else 0
        n1 = math.ceil(rc / h1) if pbc[1] else 0
        n2 = math.ceil(rc / h2) if pbc[2] else 0

        # --- Build all image shift vectors ---
        # shifts = n0 * a0 + n1 * a1 + n2 * a2 for all (n0, n1, n2)
        range0 = cp.arange(-n0, n0 + 1, dtype=cp.float64)
        range1 = cp.arange(-n1, n1 + 1, dtype=cp.float64)
        range2 = cp.arange(-n2, n2 + 1, dtype=cp.float64)

        g0, g1, g2 = cp.meshgrid(range0, range1, range2, indexing='ij')
        # (2n0+1, 2n1+1, 2n2+1) each → flatten
        n_images = g0.size
        g0 = g0.ravel()[:, None]  # (n_images, 1)
        g1 = g1.ravel()[:, None]
        g2 = g2.ravel()[:, None]

        # shifts: (n_images, 3)
        shifts = g0 * box[0] + g1 * box[1] + g2 * box[2]

        # --- Compute all pairwise distances across images ---
        # For each image s, for all (i, j): d = r_j + s - r_i
        # Use broadcasting: r_j[None, :, :] + shifts[:, None, :] - r_i[None, :, :]
        # → shape (n_images, N, N, 3) — may be large; process per image for memory

        idx_i_list = []
        idx_j_list = []
        offset_list = []

        rc_sq = rc * rc

        for s in range(n_images):
            shift = shifts[s]  # (3,)
            # dr[i, j] = r[j] - r[i] + shift, shape (N, N, 3)
            dr = r[None, :, :] - r[:, None, :] + shift[None, None, :]  # (N, N, 3)
            dist_sq = cp.sum(dr * dr, axis=2)  # (N, N)

            # Exclude self-interaction (i==j) in the zero-image cell
            is_zero_image = (g0[s, 0] == 0) and (g1[s, 0] == 0) and (g2[s, 0] == 0)
            if is_zero_image:
                cp.fill_diagonal(dist_sq, rc_sq + 1.0)

            # Find pairs within cutoff
            mask = dist_sq < rc_sq
            ii, jj = cp.where(mask)

            if len(ii) > 0:
                idx_i_list.append(ii.astype(cp.int32))
                idx_j_list.append(jj.astype(cp.int32))
                # Store the PBC image shift only (not the full displacement).
                # The force model computes r_j - r_i + offset internally.
                # 仅存储 PBC 镜像偏移（非完整位移），力场模型内部计算 r_j - r_i + offset。
                offset_list.append(
                    cp.broadcast_to(shift[None, :], (len(ii), 3)).copy()
                )

        if len(idx_i_list) == 0:
            empty = cp.array([], dtype=cp.int32)
            return empty, empty, cp.zeros((0, 3), dtype=cp.float64)

        idx_i = cp.concatenate(idx_i_list)
        idx_j = cp.concatenate(idx_j_list)
        offsets = cp.concatenate(offset_list)

        return idx_i, idx_j, offsets
