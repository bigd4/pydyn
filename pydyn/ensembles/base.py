"""Base classes for molecular dynamics integration schemes.
分子动力学积分方案的基础类。
"""
from typing import Optional, Any
import cupy as cp
from ..constants import Constants


class Ensemble:
    """Base ensemble integrator combining multiple operators.
    组合多个操作符的基础系综积分器。

    Uses operator splitting to compose elementary operators (position
    updates, force/momentum updates, thermostat/barostat operations)
    into a complete integration scheme.
    / 使用算子分裂将基本算子（位置更新、力/动量更新、
    恒温器/恒压器操作）组成完整的积分方案。
    """
    def step(self, state, context, dt):
        """Execute one integration step using operator splitting.
        使用算子分裂执行一个积分步。

        Args:
            state: Simulation state to update. / 要更新的模拟状态。
            context: Simulation context. / 模拟环境。
            dt: Timestep in picoseconds. / 时间步（皮秒）。
        """
        for op, ts in self.op_list:
            op.apply(state, context, dt * ts)
        for constraint in context.constraints:
            constraint.apply(state, context)

    def get_conserved_energy(self, state: Any, context: Any) -> Optional[float]:
        """Compute conserved energy quantity for this ensemble (optional).
        计算此系综的守恒能量量（可选）。

        For ensemble-specific integrators like Nosé-Hoover that maintain
        a specific ensemble and conserve a modified Hamiltonian, this method
        should be implemented to return the conserved quantity.
        / 对于维持特定系综并守恒修改的哈密顿量的特定系综积分器
        （如 Nosé-Hoover），应实现此方法以返回守恒量。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context. / 模拟环境。

        Returns:
            Conserved energy value, or None if not applicable.
            / 守恒能量值，或如果不适用则返回 None。
        """
        return None

    def get_pressure(self, state: Any, context: Any) -> Optional[float]:
        """Compute pressure for this ensemble (optional).
        计算此系综的压力（可选）。

        For ensemble-specific integrators that compute pressure (e.g., NPT),
        this method should be implemented to return the pressure.
        / 对于计算压力的特定系综积分器（例如 NPT），
        应实现此方法以返回压力。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context. / 模拟环境。

        Returns:
            Pressure value in appropriate units, or None if not applicable.
            / 压力值（以适当的单位），或如果不适用则返回 None。
        """
        return None


class Operator:
    """Base class for elementary operations in integration schemes.
    积分方案中基本操作的基础类。

    Operators are the building blocks of integration schemes, each performing
    a single type of update (position, momentum, thermostat, barostat, etc.).
    / 算子是积分方案的构建块，每个都执行单一类型的更新
    （位置、动量、恒温器、恒压器等）。
    """
    def apply(self, state, context, dt):
        """Apply operation to system state for timestep dt.
        在时间步dt内对系统状态应用操作。

        Args:
            state: Simulation state to modify. / 要修改的模拟状态。
            context: Simulation context. / 模拟环境。
            dt: Timestep duration. / 时间步长。

        Raises:
            NotImplementedError: If not overridden by subclass.
                / 如果未被子类重写，则引发 NotImplementedError。
        """
        raise NotImplementedError


class PositionOp(Operator):
    """Position update operator: r' = r + dt * v.
    位置更新算子：r' = r + dt * v。

    Updates atomic positions based on current momenta using
    velocity Verlet integration.
    / 使用速度 Verlet 积分根据当前动量更新原子位置。
    """
    def apply(self, state, context, dt):
        """Update positions by dt.
        按 dt 更新位置。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context. / 模拟环境。
            dt: Timestep duration. / 时间步长。
        """
        state.r += dt * state.p / state.m[:, None]


class MomentumOp(Operator):
    """Momentum update operator from forces: p' = p + dt * F.
    来自力的动量更新算子：p' = p + dt * F。

    Computes forces and updates atomic momenta using the force model.
    / 使用力模型计算力并更新原子动量。
    """
    def __init__(self, force_model):
        """Initialize with force model.
        使用力模型初始化。

        Args:
            force_model: ForceModel instance to compute forces.
                / ForceModel 实例以计算力。
        """
        self.force_model = force_model

    def apply(self, state, context, dt):
        """Compute forces and update momenta by dt.
        计算力并按 dt 更新动量。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context. / 模拟环境。
            dt: Timestep duration. / 时间步长。
        """
        self.force_model.compute(state, context)
        forces = self.force_model.results["forces"]
        state.p += dt * forces * Constants.e_to_mv2
