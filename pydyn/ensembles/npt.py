"""NPT (isotherm-isobaric) ensemble implementations.
NPT（等温等压）系综实现。
"""

from .base import Ensemble, Operator
import cupy as cp
from ..constants import Constants
from .nhc import (
    NoseHooverChainThermostatOp,
    MTTKNPTBarostatOp,
)
from scipy.special import exprel


def solve_linear_evolution(A, X, dt, b=None):
    """
    dX/dt = A X + b
    X(t+dt) = exp(A*dt) * X(t) + (I-exp(A*dt)) / A * b
            = exp(A*dt) * X(t) + exprel(A*dt) * b * dt
    """
    # Symmetrize to handle numerical asymmetry
    A = 0.5 * (A + A.T)
    eigvals, U = cp.linalg.eigh(A)
    sol = (X @ U) * cp.exp(eigvals * dt)[None, :]
    if b is not None:
        sol += dt * (b @ U) * exprel(eigvals * dt)[None, :]
    return sol @ U.T


class BoxOp(Operator):
    """Box (cell) update operator for NPT ensemble.
    用于NPT系综的晶胞更新算子。
    """

    def apply(self, state, context, dt):
        """Update cell vectors according to box momentum.
        根据晶胞动量更新晶胞矢量。
        d_box/dt = box_p / W · box
        """
        state.box = solve_linear_evolution(state.box_p / state.W, state.box, dt)


class PositionOp(Operator):
    """Position update with box deformation in NPT ensemble.
    NPT系综中带有晶胞变形的位置更新算子。
    """

    def apply(self, state, context, dt):
        """Update positions with box scaling effects.
        更新位置并考虑晶胞缩放效应。
        dr/dt = p/m + box_p / W · r
        """
        state.r = solve_linear_evolution(
            state.box_p / state.W, state.r, dt, state.p / state.m[:, None]
        )


class BoxMomentumOp(Operator):
    """Box momentum update operator for barostat.
    气压调节器的晶胞动量更新算子。
    """

    def __init__(self, force_model):
        self.force_model = force_model

    def apply(self, state, context, dt):
        """Update box momentum from stress and kinetic contributions.
        从应力和动能贡献更新晶胞动量。
        d_box_p/dt = G_stress + G_kin
        """
        self.force_model.compute(state, context)
        virial = self.force_model.results["virial"] + state.kinetic_virial

        # G_stress = V (σ − P I)
        # G_kin    = (2K / 3N) I
        # 1. virial-like stress term
        target_pV = state.volume * context.target_pressure * Constants.pV_to_e
        G_stress = virial - target_pV * cp.eye(3)  # eV

        # 2. kinetic energy term
        G_kin = 2 * state.kinetic_energy / (3 * state.N) * cp.eye(3)  # eV
        # total virial-like matrix for barostat
        state.box_p += dt * (G_stress + G_kin)  # [eV*ps]


class MomentumOp(Operator):
    """Atomic momentum update operator for NPT ensemble.
    NPT系综的原子动量更新算子。
    """

    def __init__(self, force_model):
        self.force_model = force_model

    def apply(self, state, context, dt):
        """Update atomic momenta with box deformation coupling.
        更新原子动量并考虑晶胞变形耦合。
        dp/dt = F - (box_p + Tr(box_p)*I/3N)/W · p
        """
        self.force_model.compute(state, context)
        forces = self.force_model.results["forces"]

        state.p = solve_linear_evolution(
            -(state.box_p + cp.trace(state.box_p) / (3 * state.N) * cp.eye(3))
            / state.W,
            state.p,
            dt,
            Constants.e_to_mv2 * forces,
        )


class MTTKNPT(Ensemble):
    """NPT ensemble using MTTK (Martyna-Tobias-Tuckerman-Klein) algorithm.
    使用MTTK算法的NPT系综。
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
            (PositionOp(), 1.0),
            (BoxOp(), 1.0),
            (MomentumOp(force_model), 0.5),
            (BoxMomentumOp(force_model), 0.5),
            (NoseHooverChainThermostatOp(t_tau), 0.5),
            (MTTKNPTBarostatOp(p_tau), 0.5),
        ]

    def get_conserved_energy(self, state, context):
        """Calculate conserved quantity for MTTK NPT ensemble.
        计算MTTK NPT系综的守恒量。
        """

        kT = Constants.kB * context.target_temp  # eV
        self.force_model.compute(state, context)
        potential_energy = self.force_model.results["potential_energy"]
        barostat_energy = (
            cp.sum(state.p_xi**2 / state.R) / 2
            + 9 * kT * state.xi[0]
            + kT * cp.sum(state.xi[1:])
        )
        thermostat_energy = (
            3 * state.N * kT * state.eta[0]
            + kT * cp.sum(state.eta[1:])
            + cp.sum(0.5 * state.p_eta**2 / state.Q)
        )
        conserved_energy = (
            potential_energy
            + state.kinetic_energy
            + barostat_energy
            + thermostat_energy
            + cp.trace(state.box_p.T @ state.box_p) / (2 * state.W)
            + context.target_pressure * state.volume * Constants.pV_to_e
        )
        return float(conserved_energy)

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
