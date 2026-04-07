"""Simulation context and thermodynamic parameters.
模拟环境和热力学参数。

This module defines the SimulationContext class that holds ensemble
specifications and constraints for molecular dynamics simulations.
"""

from typing import Optional, List, Any
from .constants import Constants


class SimulationContext:
    """Container for simulation ensemble parameters and constraints.
    模拟系综参数和约束的容器。

    Stores the target thermodynamic conditions (temperature, pressure) and
    constraint specifications for the simulation.
    / 存储模拟的目标热力学条件（温度、压力）和约束规范。

    Attributes:
        target_temp: Target temperature in Kelvin for thermostats.
            / 恒温器的目标温度（开尔文）。
        target_pressure: Target pressure in bar for barostats.
            / 恒压器的目标压力（巴）。
        constraints: List of constraint objects to enforce.
            / 要执行的约束对象列表。
    """

    def __init__(
        self,
        target_temp: Optional[float] = None,
        target_pressure: Optional[float] = None,
        constraints: Optional[List[Any]] = None,
    ) -> None:
        """Initialize a simulation context.
        初始化模拟环境。

        Args:
            target_temp: Target temperature in Kelvin.
                / 目标温度（开尔文）。
            target_pressure: Target pressure in bar. / 目标压力（巴）。
            constraints: List of Constraint objects to apply.
                / 要应用的约束对象的列表。
        """
        self.target_temp: Optional[float] = target_temp
        self.target_pressure: Optional[float] = target_pressure
        self.constraints: List[Any] = constraints if constraints is not None else []

    def get_temperature(self, state: Any) -> float:
        """Compute instantaneous temperature from kinetic energy.
        从动能计算瞬时温度。

        Uses equipartition theorem: T = 2*KE / (dof * k_B), where dof is
        the number of degrees of freedom after constraints. Returns 0.0 if
        no degrees of freedom are available.
        / 使用等分配定理：T = 2*KE / (dof * k_B)，其中 dof 是约束后的自由度数。
        如果没有可用的自由度，则返回 0.0。

        Args:
            state: Simulation state with momentum and mass data.
                / 具有动量和质量数据的模拟状态。

        Returns:
            Instantaneous temperature in Kelvin.
            / 瞬时温度（开尔文）。
        """
        dof = 3 * state.N - sum(c.removed_dof for c in self.constraints)
        # Avoid division by zero for single-atom systems with no degrees of freedom
        if dof <= 0:
            return 0.0
        return 2 * state.kinetic_energy / (dof * Constants.kB)
