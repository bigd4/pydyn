"""Constraints for molecular dynamics simulations.
分子动力学模拟的约束。

This module provides constraint classes that modify the system to enforce
physical or numerical constraints during simulation.
"""

from typing import Any
import cupy as cp
from .constants import Constants


class Constraint:
    """Base class for simulation constraints.
    模拟约束的基类。

    Constraints modify the system state to enforce specific conditions,
    such as removing center-of-mass motion or fixing atomic positions.
    / 约束修改系统状态以执行特定条件，例如移除质心运动或固定原子位置。

    Attributes:
        removed_dof: Number of degrees of freedom removed by this constraint.
            / 此约束移除的自由度数。Used in temperature calculations.
            / 用于温度计算。
    """

    removed_dof: int = 0
    """Degrees of freedom removed. / 移除的自由度数。"""

    def apply(self, state: Any, context: Any) -> None:
        """Apply the constraint to the simulation state.
        将约束应用于模拟状态。

        Args:
            state: Simulation state to modify. / 要修改的模拟状态。
            context: Simulation context. / 模拟环境。

        Raises:
            NotImplementedError: This is an abstract method.
                / 这是一个抽象方法。
        """
        raise NotImplementedError


class RemoveCOMMomentum(Constraint):
    """Constraint that removes center-of-mass momentum.
    移除质心动量的约束。

    Ensures the total momentum of the system equals zero by rescaling
    individual atomic momenta. This removes 3 degrees of freedom from
    the system (one for each Cartesian direction).
    / 通过重新缩放单个原子动量来确保系统的总动量为零。
    这从系统中移除 3 个自由度（每个笛卡尔方向一个）。
    """

    removed_dof: int = 3
    """This constraint removes 3 translational degrees of freedom.
    / 此约束移除 3 个平移自由度。"""

    def apply(self, state: Any, context: Any) -> None:
        """Remove center-of-mass momentum from the system.
        从系统中移除质心动量。

        Computes the center-of-mass velocity from total momentum and mass,
        then subtracts the corresponding momentum from each atom.
        / 从总动量和质量计算质心速度，然后从每个原子的动量中减去相应的动量。

        Args:
            state: Simulation state to modify.
                / 要修改的模拟状态。Momenta will be rescaled.
                / 动量将被重新缩放。
            context: Simulation context. / 模拟环境。
        """
        v_com = cp.sum(state.p, axis=0) / cp.sum(state.m)
        state.p -= state.m[:, None] * v_com
