"""Thermodynamic property calculations for simulations.
模拟的热力学性质计算。

This module provides the Thermal class for computing and monitoring
thermodynamic properties during molecular dynamics simulations.
"""

from typing import Any
import cupy as cp
from .constants import Constants


class Thermal:
    """Compute thermodynamic properties during a simulation.
    在模拟期间计算热力学性质。

    Provides convenient access to thermodynamic quantities such as
    kinetic energy and temperature by querying the simulation state
    and context.
    / 通过查询模拟状态和环境，提供对动能和温度等热力学量的便捷访问。

    Attributes:
        sim: Simulation object to monitor. / 要监视的模拟对象。
    """

    def __init__(self, sim: Any) -> None:
        """Initialize the thermal property calculator.
        初始化热力学性质计算器。

        Args:
            sim: Simulation object with state and context.
                / 具有状态和环境的模拟对象。
        """
        self.sim: Any = sim

    @property
    def kinetic_energy(self) -> float:
        """Get the current kinetic energy of the system.
        获取系统的当前动能。

        Returns:
            Kinetic energy in eV. / 动能（eV）。
        """
        return self.sim.state.kinetic_energy

    @property
    def temperature(self) -> float:
        """Get the current instantaneous temperature.
        获取当前的瞬时温度。

        Computed from the kinetic energy and degrees of freedom,
        taking into account any active constraints.
        / 从动能和自由度计算，考虑任何活跃的约束。

        Returns:
            Instantaneous temperature in Kelvin.
            / 瞬时温度（开尔文）。
        """
        return self.sim.context.get_temperature(self.sim.state)
