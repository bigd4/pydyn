"""Velocity and configuration initializers for molecular dynamics.
分子动力学的速度和配置初始化程序。

This module provides initializer classes for setting up atomic velocities
and configurations according to statistical mechanical distributions.
"""

from typing import Optional, Any
import cupy as cp
from .constants import Constants


class VelocityInitializer:
    """Base class for velocity initialization strategies.
    速度初始化策略的基类。

    Defines the interface for initializing atomic velocities and momenta
    according to different statistical distributions.
    / 定义根据不同统计分布初始化原子速度和动量的接口。
    """

    def initialize(self, state: Any, context: Any) -> None:
        """Initialize velocities/momenta in the given state.
        初始化给定状态中的速度/动量。

        Args:
            state: Simulation state to initialize.
                / 要初始化的模拟状态。Momenta will be set.
                / 动量将被设置。
            context: Simulation context with target parameters.
                / 具有目标参数的模拟环境。

        Raises:
            NotImplementedError: This is an abstract method.
                / 这是一个抽象方法。
        """
        raise NotImplementedError


class MaxwellBoltzmannDistribution(VelocityInitializer):
    """Velocity initializer using Maxwell-Boltzmann distribution.
    使用 Maxwell-Boltzmann 分布的速度初始化程序。

    Generates atomic velocities (momenta) from a Maxwell-Boltzmann
    distribution at a specified temperature. This is the standard choice
    for equilibrium molecular dynamics simulations.
    / 从指定温度下的 Maxwell-Boltzmann 分布生成原子速度（动量）。
    这是平衡分子动力学模拟的标准选择。

    Attributes:
        target_temp: Temperature for distribution; if None, uses
            context.target_temp. / 分布的温度；如果为 None，则使用 context.target_temp。
    """

    def __init__(self, target_temp: Optional[float] = None) -> None:
        """Initialize the Maxwell-Boltzmann velocity initializer.
        初始化 Maxwell-Boltzmann 速度初始化程序。

        Args:
            target_temp: Target temperature in Kelvin; defaults to None
                to use context.target_temp. / 目标温度（开尔文）；默认为 None
                以使用 context.target_temp。
        """
        self.target_temp: Optional[float] = target_temp

    def initialize(self, state: Any, context: Any) -> None:
        """Initialize momenta from Maxwell-Boltzmann distribution.
        从 Maxwell-Boltzmann 分布初始化动量。

        Generates momenta p_i from independent Gaussian distributions with
        variance sigma_p = sqrt(m_i * k_B * T * e_to_mv2), ensuring the
        initial kinetic energy corresponds to the target temperature.
        / 从方差为 sigma_p = sqrt(m_i * k_B * T * e_to_mv2) 的独立高斯分布
        生成动量 p_i，确保初始动能对应于目标温度。

        Args:
            state: Simulation state to initialize.
                / 要初始化的模拟状态。Momenta will be set.
                / 动量将被设置。
            context: Simulation context with target thermodynamic parameters.
                / 具有目标热力学参数的模拟环境。
        """
        target_temp = self.target_temp or context.target_temp
        sigma_p = cp.sqrt(
            state.m[:, None] * Constants.kB * target_temp * Constants.e_to_mv2
        )
        state.p = cp.random.standard_normal((state.N, 3)) * sigma_p
