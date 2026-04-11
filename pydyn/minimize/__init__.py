"""Structural optimization (geometry minimization) module.
结构优化（几何弛豫）模块。

Uses a Filter abstraction to expose any subset of degrees of freedom
(atomic positions, cell vectors, spin orientations, or combinations)
as a flat array that an optimizer can drive toward an energy minimum.
/ 使用 Filter 抽象将任意自由度子集（原子位置、晶胞矢量、自旋取向或其组合）
暴露为优化器可驱动至能量最小值的平坦数组。

Typical usage:
    from pydyn.minimize import FIRE, AtomFilter, SpinFilter, CompositeFilter, Minimization

    # Atoms + spins simultaneously (single force_model.compute per step)
    filter = CompositeFilter(state, [AtomFilter(state), SpinFilter(state)])
    opt = FIRE(filter)
    driver = Minimization(opt, force_model, context)
    driver.run(max_steps=1000, fmax=0.05)
"""

from .filters import Filter, AtomFilter, CellFilter, SpinFilter, CompositeFilter
from .fire import FIRE
from .driver import Minimization

__all__ = [
    "Filter",
    "AtomFilter",
    "CellFilter",
    "SpinFilter",
    "CompositeFilter",
    "FIRE",
    "Minimization",
]
