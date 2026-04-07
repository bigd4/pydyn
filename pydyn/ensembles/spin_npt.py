"""Spin dynamics with NPT (isothermal-isobaric) ensemble.
自旋动力学与NPT（等温等压）系综。
"""
from .base import Ensemble
import cupy as cp
from ..constants import Constants
from .nhc import (
    NoseHooverChainThermostatOp,
    MTTKNPTBarostatOp,
)
from .npt import BoxMomentumOp, MomentumOp, PositionOp, BoxOp
from .spin import SIBSpinOp


class SpinMTTKNPT(Ensemble):
    """NPT ensemble for magnetic systems with MTTK barostat.
    用于磁性系统的MTTK气压调节器NPT系综。
    """
    def __init__(self, t_tau, p_tau, force_model):
        self.force_model = force_model
        self.t_tau = t_tau
        self.p_tau = p_tau
        self.op_list = [
            (MTTKNPTBarostatOp(p_tau), 0.5),
            (NoseHooverChainThermostatOp(t_tau), 0.5),
            (BoxMomentumOp(force_model), 0.5),
            (MomentumOp(force_model), 0.5),
            (SIBSpinOp(force_model), 0.5),
            (PositionOp(), 1.0),
            (BoxOp(), 1.0),
            (SIBSpinOp(force_model), 0.5),
            (MomentumOp(force_model), 0.5),
            (BoxMomentumOp(force_model), 0.5),
            (NoseHooverChainThermostatOp(t_tau), 0.5),
            (MTTKNPTBarostatOp(p_tau), 0.5),
        ]

    def get_pressure(self, state, context):
        """Calculate instantaneous pressure from virial.
        从维里定理计算瞬时压力。
        """
        self.force_model.compute(state, context)
        virial = self.force_model.results["virial"]
        kinetic_virial = (
            state.p.T @ (state.p / state.m[:, None]) * Constants.mv2_to_e
        )  # eV
        total_virial = virial + kinetic_virial
        pressure = (
            cp.trace(total_virial) / (3 * state.volume * Constants.pV_to_e) / 10000
        )  # in GPa
        return float(pressure)

    def get_spin_temperature(self, state, context):
        """Calculate effective temperature from spin dynamics.
        从自旋动力学计算有效温度。
        """
        self.force_model.compute(state, context, properties=["spin_torques"])
        B_eff = self.force_model.results["spin_torques"]
        spin_temp = (
            cp.sum(cp.linalg.norm(cp.cross(state.spin.spins, B_eff), axis=1) ** 2)
            / cp.sum(state.spin.spins * B_eff)
            / (2 * Constants.kB)
        )
        return spin_temp
