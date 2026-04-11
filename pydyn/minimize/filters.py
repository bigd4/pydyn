"""Filters that expose degrees of freedom as flat position/force arrays.
将自由度暴露为平坦位置/力数组的过滤器。

Each Filter wraps a State and projects a subset of its DOF into a
unified (positions, forces) interface that optimizers can operate on.
/ 每个 Filter 封装一个 State，将其部分自由度投影到统一的
（位置，力）接口，供优化器操作。

Reference for UnitCellFilter: Knuth et al., J. Chem. Phys. 143, 102601 (2015)
Reference for spin tangent-space projection: same convention as SIBSpinOp
"""

import cupy as cp
from typing import List


class Filter:
    """Base class exposing a subset of state DOF to an optimizer.
    向优化器暴露状态部分自由度的基类。

    Subclasses must implement get_positions, set_positions, extract_forces,
    and n_dof. The state attribute must be set by the subclass.
    / 子类必须实现 get_positions、set_positions、extract_forces 和 n_dof。
    子类必须设置 state 属性。
    """

    required_properties: List[str] = []

    def get_positions(self) -> cp.ndarray:
        """Return current DOF values as a flat array.
        将当前自由度值作为平坦数组返回。

        Returns:
            1D array of DOF values. / 自由度值的一维数组。
        """
        raise NotImplementedError

    def set_positions(self, x: cp.ndarray) -> None:
        """Write a flat array back into the state's DOF.
        将平坦数组写回状态的自由度。

        Args:
            x: 1D array of DOF values. / 自由度值的一维数组。
        """
        raise NotImplementedError

    def extract_forces(self, results: dict) -> cp.ndarray:
        """Extract and transform the relevant forces from computed results.
        从计算结果中提取并变换相关力。

        Called by get_forces / CompositeFilter after force_model.compute().
        / 在 force_model.compute() 之后由 get_forces / CompositeFilter 调用。

        Args:
            results: force_model.results dict. / 力场模型结果字典。

        Returns:
            1D force array matching the DOF layout of get_positions.
            / 与 get_positions 自由度布局匹配的一维力数组。
        """
        raise NotImplementedError

    def n_dof(self) -> int:
        """Number of degrees of freedom exposed by this filter.
        此过滤器暴露的自由度数量。
        """
        raise NotImplementedError

    def get_forces(self, force_model, context) -> cp.ndarray:
        """Compute forces for this filter's DOF.
        计算此过滤器自由度的力。

        Calls force_model.compute then extract_forces. For simultaneous
        optimization of multiple DOF types, prefer CompositeFilter which
        calls compute only once.
        / 调用 force_model.compute 然后调用 extract_forces。对于同时优化
        多种自由度类型，推荐使用 CompositeFilter，它只调用一次 compute。

        Args:
            force_model: Force model instance. / 力场模型实例。
            context: Simulation context. / 模拟环境。

        Returns:
            1D force array. / 一维力数组。
        """
        force_model.compute(self.state, context, properties=self.required_properties)
        return self.extract_forces(force_model.results)

    def converged(self, force_model, context, fmax: float) -> bool:
        """Check if maximum force component is below fmax.
        检查最大力分量是否低于 fmax。

        Args:
            force_model: Force model instance. / 力场模型实例。
            context: Simulation context. / 模拟环境。
            fmax: Force convergence threshold. / 力收敛阈值。

        Returns:
            True if converged. / 如果收敛则为 True。
        """
        return bool(cp.max(cp.abs(self.get_forces(force_model, context))) < fmax)


class AtomFilter(Filter):
    """Exposes Cartesian atomic positions as DOF.
    将笛卡尔原子位置作为自由度暴露。

    Positions: r.ravel() in Angstroms, shape (3N,).
    Forces   : force_model.results["forces"].ravel() in eV/Ang, shape (3N,).
    / 位置：r.ravel()（埃），形状 (3N,)。
    / 力：force_model.results["forces"].ravel()（eV/Å），形状 (3N,)。
    """

    required_properties = ["forces"]

    def __init__(self, state):
        self.state = state

    def get_positions(self) -> cp.ndarray:
        return self.state.r.ravel()

    def set_positions(self, x: cp.ndarray) -> None:
        self.state.r = x.reshape(-1, 3)

    def extract_forces(self, results: dict) -> cp.ndarray:
        return results["forces"].ravel()

    def n_dof(self) -> int:
        return self.state.N * 3


class CellFilter(Filter):
    """Exposes cell vectors as DOF with forces derived from the virial tensor.
    将晶胞矢量作为自由度暴露，力由维里张量推导。

    Positions: box.ravel() in Angstroms, shape (9,).
    Forces   : virial @ inv(box).T, shape (9,), units eV/Ang.

    The derivation follows dE/dh = -virial @ h^{-T} for zero target pressure,
    so the force (negative gradient) is virial @ inv(box).T.
    / 位置：box.ravel()（埃），形状 (9,)。
    / 力：virial @ inv(box).T，形状 (9,)，单位 eV/Å。
    / 推导遵循零目标压力下 dE/dh = -virial @ h^{-T}，
    / 因此力（负梯度）为 virial @ inv(box).T。

    Note: atomic positions are kept fixed in Cartesian space. Pair this with
    AtomFilter in a CompositeFilter to relax both simultaneously.
    / 注意：原子位置保持笛卡尔坐标不变。与 CompositeFilter 中的
    AtomFilter 配合使用可同时弛豫两者。
    """

    required_properties = ["virial"]

    def __init__(self, state):
        self.state = state

    def get_positions(self) -> cp.ndarray:
        return self.state.box.ravel()

    def set_positions(self, x: cp.ndarray) -> None:
        self.state.box = x.reshape(3, 3)

    def extract_forces(self, results: dict) -> cp.ndarray:
        virial = results["virial"]             # (3,3) eV
        inv_box = cp.linalg.inv(self.state.box)  # (3,3) Ang^{-1}
        # Force = virial @ inv(box).T  →  units: eV/Ang, matching atomic forces
        return (virial @ inv_box.T).ravel()

    def n_dof(self) -> int:
        return 9


class SpinFilter(Filter):
    """Exposes spin orientations as DOF with tangent-space projected forces.
    将自旋取向作为自由度暴露，力经切空间投影。

    Positions: spin.vector.ravel() (unit vectors), shape (3N,).
    Forces   : tangent projection of B_eff onto the sphere: B - (S·B)S,
               scaled by mu, shape (3N,).

    The spin magnitude is preserved: set_positions renormalises vectors
    so |S| stays constant throughout optimisation.
    / 位置：spin.vector.ravel()（单位向量），形状 (3N,)。
    / 力：B_eff 在球面切空间的投影：B - (S·B)S，乘以 mu，形状 (3N,)。
    / 自旋大小保持不变：set_positions 重新归一化向量，
    / 使 |S| 在整个优化过程中保持恒定。
    """

    required_properties = ["spin_torques"]

    def __init__(self, state, mu: float = 1.0):
        """
        Args:
            state: Simulation state with SpinExtension. / 带有 SpinExtension 的模拟状态。
            mu: Scale factor for spin forces relative to atomic forces.
                Tune this to balance convergence rates between DOF types.
                / 自旋力相对于原子力的缩放因子。
                / 调整此参数以平衡不同自由度类型之间的收敛速率。
        """
        self.state = state
        self.mu = mu

    def get_positions(self) -> cp.ndarray:
        return self.state.spin.vector.ravel()

    def set_positions(self, x: cp.ndarray) -> None:
        vecs = x.reshape(self.state.N, 3)
        norms = cp.linalg.norm(vecs, axis=1, keepdims=True)
        self.state.spin.vector = vecs / cp.maximum(norms, 1e-10)

    def extract_forces(self, results: dict) -> cp.ndarray:
        B_eff = results["spin_torques"]    # (N,3) eV
        s = self.state.spin.vector         # (N,3) unit vectors
        # Tangent-space projection: remove the component along s
        s_dot_B = cp.sum(s * B_eff, axis=1, keepdims=True)
        return ((B_eff - s_dot_B * s) * self.mu).ravel()

    def n_dof(self) -> int:
        return self.state.N * 3


class CompositeFilter:
    """Combines multiple filters into a single unified DOF space.
    将多个过滤器合并为单一统一的自由度空间。

    force_model.compute is called exactly once per get_forces call,
    collecting all required properties in a single pass.
    / 每次调用 get_forces 时恰好调用一次 force_model.compute，
    在单次传递中收集所有所需属性。

    Example:
        # Relax atoms + cell + spins simultaneously
        # 同时弛豫原子、晶胞和自旋
        opt = FIRE(CompositeFilter(state, [
            AtomFilter(state),
            CellFilter(state),
            SpinFilter(state),
        ]))
    """

    def __init__(self, state, filters: List[Filter]):
        """
        Args:
            state: Simulation state shared by all filters.
                / 所有过滤器共享的模拟状态。
            filters: List of Filter instances. Each must reference the same state.
                / Filter 实例列表。每个实例必须引用相同的状态。
        """
        self.state = state
        self.filters = filters

    @property
    def required_properties(self) -> List[str]:
        """Union of required_properties from all child filters.
        所有子过滤器的 required_properties 的并集。
        """
        return list({p for f in self.filters for p in f.required_properties})

    def get_positions(self) -> cp.ndarray:
        """Concatenate positions from all filters.
        连接所有过滤器的位置。
        """
        return cp.concatenate([f.get_positions() for f in self.filters])

    def set_positions(self, x: cp.ndarray) -> None:
        """Dispatch position chunks to each filter.
        将位置块分发给每个过滤器。
        """
        offset = 0
        for f in self.filters:
            n = f.n_dof()
            f.set_positions(x[offset:offset + n])
            offset += n

    def get_forces(self, force_model, context) -> cp.ndarray:
        """Compute forces with a single force_model.compute call.
        通过单次 force_model.compute 调用计算力。

        Args:
            force_model: Force model instance. / 力场模型实例。
            context: Simulation context. / 模拟环境。

        Returns:
            Concatenated 1D force array from all filters.
            / 所有过滤器的连接一维力数组。
        """
        force_model.compute(
            self.state, context, properties=self.required_properties
        )
        return cp.concatenate(
            [f.extract_forces(force_model.results) for f in self.filters]
        )

    def converged(self, force_model, context, fmax: float) -> bool:
        """Check if max force component across all DOF is below fmax.
        检查所有自由度中的最大力分量是否低于 fmax。

        Args:
            force_model: Force model instance. / 力场模型实例。
            context: Simulation context. / 模拟环境。
            fmax: Convergence threshold. / 收敛阈值。

        Returns:
            True if all DOF are converged. / 如果所有自由度都收敛则为 True。
        """
        return bool(
            cp.max(cp.abs(self.get_forces(force_model, context))) < fmax
        )

    def n_dof(self) -> int:
        """Total number of degrees of freedom across all filters.
        所有过滤器的自由度总数。
        """
        return sum(f.n_dof() for f in self.filters)
