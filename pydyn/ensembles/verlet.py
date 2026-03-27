"""Velocity Verlet integration scheme for NVE ensemble.
用于NVE系综的速度Verlet积分方案。
"""
from .base import PositionOp, MomentumOp, Ensemble


class VelocityVerlet(Ensemble):
    """Velocity Verlet integrator: symplectic, time-reversible, NVE ensemble.
    速度Verlet积分器：辛结构、时间可逆、NVE系综。
    """
    def __init__(self, force_model):
        self.op_list = [
            (MomentumOp(force_model), 0.5),
            (PositionOp(), 1.0),
            (MomentumOp(force_model), 0.5),
        ]
