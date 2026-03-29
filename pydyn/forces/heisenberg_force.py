"""Heisenberg exchange interaction force model for spin systems.
自旋系统的海森堡交换相互作用力模型。

The Heisenberg model describes magnetic interactions between localized spins
on a lattice. It is widely used for studying ferromagnetic and antiferromagnetic
materials.

海森堡模型描述了晶格上定域自旋之间的磁相互作用。它被广泛用于研究
铁磁和反铁磁材料。

Reference:
    Heisenberg, W. (1928). Zur Theorie des Ferromagnetismus. 
    Zeitschrift für Physik, 49(9-10), 619-636.
"""

import cupy as cp
from typing import Optional, List, Tuple
from .base import ForceModel


class HeisenbergForceModel(ForceModel):
    """Heisenberg exchange interaction force model.
    海森堡交换相互作用力模型。
    
    The Hamiltonian for the Heisenberg model is:
    H = -J * Σ_{<i,j>} S_i · S_j
    
    where:
    - J is the exchange coupling constant (J > 0: ferromagnetic, J < 0: antiferromagnetic)
    - S_i is the spin vector (unit vector) of atom i
    - <i,j> denotes summation over nearest neighbor pairs
    
    The effective magnetic field (spin torque) on atom i is:
    B_i = J * Σ_j S_j
    
    海森堡模型的哈密顿量为：
    H = -J * Σ_{<i,j>} S_i · S_j
    
    其中：
    - J 是交换耦合常数（J > 0：铁磁，J < 0：反铁磁）
    - S_i 是原子 i 的自旋向量（单位向量）
    - <i,j> 表示对最近邻对求和
    
    原子 i 上的有效磁场（自旋扭矩）为：
    B_i = J * Σ_j S_j
    
    Attributes:
        J: Exchange coupling constant in eV. / 交换耦合常数（eV）。
        cutoff: Neighbor cutoff distance in Angstroms. / 近邻截断距离（埃）。
        neighbor_list: Optional pre-computed neighbor list. / 可选的预计算近邻列表。
    """
    
    implemented_properties = ["spin_torques", "energy", "magnetic_energy"]
    
    def __init__(
        self,
        J: float = 1.0,
        cutoff: float = 3.0,
        neighbor_list=None
    ):
        """Initialize Heisenberg force model.
        初始化海森堡力模型。
        
        Args:
            J: Exchange coupling constant in eV. Positive for ferromagnetic
                coupling, negative for antiferromagnetic. Default is 1.0 eV.
                / 交换耦合常数（eV）。正值为铁磁耦合，负值为反铁磁耦合。
            cutoff: Neighbor cutoff distance in Angstroms. Default is 3.0.
                / 近邻截断距离（埃）。
            neighbor_list: Optional neighbor list calculator. If None, a simple
                distance-based neighbor search is used.
                / 可选的近邻列表计算器。如果为 None，则使用简单的基于距离的近邻搜索。
        """
        super().__init__()
        self.J = J
        self.cutoff = cutoff
        self.neighbor_list = neighbor_list
    
    def compute(self, state, context, properties: Optional[List[str]] = None):
        """Compute spin torques and energy for Heisenberg model.
        计算海森堡模型的自旋扭矩和能量。
        
        Args:
            state: Current simulation state with spin extension.
                / 当前模拟状态（需要包含自旋扩展）。
            context: Simulation context. / 模拟环境。
            properties: List of properties to compute. If None, computes all
                implemented properties. / 要计算的属性列表。
        
        Raises:
            AttributeError: If state does not have spin extension.
                / 如果状态没有自旋扩展。
        """
        if properties is None:
            properties = self.implemented_properties
        
        if not self.need_compute(state, context, properties):
            return
        
        # Check that state has spin extension
        if not hasattr(state, 'spin'):
            raise AttributeError(
                "State must have spin extension for Heisenberg model. "
                "Use SpinExtension when creating the state."
            )
        
        # Get spin vectors (unit vectors)
        spins = state.spin.vector  # shape (N, 3)
        N = state.N
        
        # Build or use neighbor list
        if self.neighbor_list is not None:
            idx_i, idx_j, offsets = self.neighbor_list.find_neighbor(state)
            # Filter by cutoff
            mask = cp.linalg.norm(offsets, axis=1) < self.cutoff
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
        else:
            idx_i, idx_j = self._build_simple_neighbor_list(state)
        
        # Compute effective magnetic field (spin torque): B_i = J * Σ_j S_j
        # 计算有效磁场（自旋扭矩）
        spin_torques = cp.zeros((N, 3), dtype=cp.float64)
        
        if len(idx_i) > 0:
            # Vectorized neighbor sum
            # 向量化近邻求和
            for i in range(N):
                neighbors = idx_j[idx_i == i]
                if len(neighbors) > 0:
                    spin_torques[i] = self.J * cp.sum(spins[neighbors], axis=0)
        
        # Compute total energy: E = -J * Σ_{<i,j>} S_i · S_j
        # Each pair (i,j) with i < j is counted once
        # 计算总能量
        energy = 0.0
        if len(idx_i) > 0:
            dot_products = cp.sum(spins[idx_i] * spins[idx_j], axis=1)
            energy = -self.J * cp.sum(dot_products)
        
        self.results["spin_torques"] = spin_torques
        self.results["energy"] = energy
        self.results["magnetic_energy"] = energy
        self.state = state.copy()
    
    def _build_simple_neighbor_list(self, state) -> Tuple[cp.ndarray, cp.ndarray]:
        """Build simple neighbor list based on distance cutoff.
        基于距离截断构建简单近邻列表。
        
        This is a simple O(N^2) implementation suitable for small systems.
        For larger systems, consider using CudaNeighborList.
        
        这是一个简单的 O(N^2) 实现，适用于小系统。对于大系统，请考虑使用
        CudaNeighborList。
        
        Args:
            state: Simulation state with positions. / 包含位置的模拟状态。
        
        Returns:
            Tuple of (idx_i, idx_j) arrays for neighbor pairs with i < j.
            邻居对 (i,j) 的索引数组元组，其中 i < j。
        """
        N = state.N
        
        idx_i_list = []
        idx_j_list = []
        
        cutoff_sq = self.cutoff ** 2
        
        for i in range(N):
            for j in range(i + 1, N):  # Only i < j to avoid double counting
                # Calculate distance with periodic boundary conditions
                dr = state.r[j] - state.r[i]
                
                # Apply minimum image convention for PBC
                if state.pbc is not None and state.box is not None:
                    for dim in range(3):
                        if state.pbc[dim]:
                            box_len = state.box[dim, dim]
                            dr[dim] -= box_len * cp.round(dr[dim] / box_len)
                
                dist_sq = cp.sum(dr ** 2)
                
                if dist_sq < cutoff_sq:
                    idx_i_list.append(i)
                    idx_j_list.append(j)
        
        if len(idx_i_list) == 0:
            return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)
        
        return cp.array(idx_i_list, dtype=cp.int32), cp.array(idx_j_list, dtype=cp.int32)
