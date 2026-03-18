# https://www.lammps.org/workshops/Aug17/pdf/tranchida.pdf

import numpy as np
from ase import Atoms
from typing import Dict
import types
from ase.io import read, write
import torch
import cupy as cp
from ase.data import atomic_numbers
from .base import Ensemble, Operator
import cupy as cp
from ..constants import Constants
from .nhc import (
    NoseHooverChainThermostatOp,
    MTTKNPTBarostatOp,
)
from .npt import BoxMomentumOp, MomentumOp, PositionOp, BoxOp
from copy import deepcopy


def sib_transform(spins, omegas):
    """
    DOI: 10.1103/PhysRevB.99.224414
    Perform the SIB (Semi-Implicit B) spin update.
    n^p_i = n_i - (n_i + n^p_i) / 2 x (omegas_i)
    """
    A = 0.5 * omegas  # shape (n,3)
    detAi = 1.0 / (1.0 + cp.sum(A**2, axis=1))  # shape (n)
    a2 = spins - cp.cross(spins, A)  # predictor, shape (n,3)

    # Precompute repeated terms
    A0, A1, A2 = A[:, 0], A[:, 1], A[:, 2]
    a20, a21, a22 = a2[:, 0], a2[:, 1], a2[:, 2]

    out = cp.zeros_like(spins)
    out[:, 0] = a20 * (A0 * A0 + 1) + a21 * (A0 * A1 - A2) + a22 * (A0 * A2 + A1)
    out[:, 1] = a20 * (A1 * A0 + A2) + a21 * (A1 * A1 + 1) + a22 * (A1 * A2 - A0)
    out[:, 2] = a20 * (A2 * A0 - A1) + a21 * (A2 * A1 + A0) + a22 * (A2 * A2 + 1)
    out *= detAi[:, None]
    return out


class SIBSpinOp(Operator):

    def __init__(
        self,
        force_model,
        alpha_t: float = 0.1,
    ):
        self.force_model = force_model
        self.alpha_t = alpha_t
        llg_damping_norm = 1 + self.alpha_t**2
        e_to_omega = Constants.gamma / Constants.mu_B  # [(rad/ps) / eV]

        # deterministic spin precession
        self.drift_base = e_to_omega / llg_damping_norm  # no mu_i and dt

        # stochastic thermal noise
        self.noise_base = (
            cp.sqrt(2 * self.alpha_t * e_to_omega) / llg_damping_norm
        )  # no mu_i, kT, and dt

    def apply(self, state, context, dt):
        """
        sp_t - s_t = (sp_t + s_t) / 2 * omega
        s_t+1 - s_t = (s_t + s_t+1) / 2 * omega
        omega = dt * (B_eff + alpha * s x B_eff) + sqrt(dt) * noise
        """
        kT = Constants.kB * context.target_temp
        mu_i_inv = cp.where(state.mu_i > 0.01, 1.0 / state.mu_i, 0.0)
        self.drift_prefactor = self.drift_base * mu_i_inv * dt
        self.noise_prefactor = self.noise_base * cp.sqrt(mu_i_inv * kT * dt)

        noise = cp.random.randn(len(state.spins), 3)
        omega = self.calc_omega(state, context, noise)
        spins_predict = sib_transform(state.spins, omega)
        state_p = state.copy()
        state_p.spins = (spins_predict + state.spins) / 2
        omega = self.calc_omega(state_p, context, noise)
        state.spins = sib_transform(state.spins, omega)

    def calc_omega(self, state, context, noise):

        self.force_model.compute(state, context, properties=["spin_torques"])
        B_eff = self.force_model.results["spin_torques"]
        omega = (
            self.drift_prefactor[:, None]
            * (B_eff + cp.cross(state.spins, B_eff) * self.alpha_t)
            + self.noise_prefactor[:, None] * noise
        )
        return omega


class SpinMTTKNPT(Ensemble):

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
        self.force_model.compute(state, context, properties=["spin_torques"])
        B_eff = self.force_model.results["spin_torques"]
        spin_temp = (
            cp.sum(cp.linalg.norm(cp.cross(state.spins, B_eff), axis=1) ** 2)
             / cp.sum(state.spins * B_eff)
             / (2 * Constants.kB)
        )
        return spin_temp

class SpinGoGoGo(Ensemble):

    def __init__(self, force_model):
        self.force_model = force_model
        self.op_list = [
            (SIBSpinOp(force_model), 1.0),
        ]

    def get_spin_temperature(self, state, context):
        self.force_model.compute(state, context, properties=["spin_torques"])
        B_eff = self.force_model.results["spin_torques"]
        spin_temp = (
            cp.sum(cp.linalg.norm(cp.cross(state.spins, B_eff), axis=1) ** 2)
             / cp.sum(state.spins * B_eff)
             / (2 * Constants.kB)
        )
        return spin_temp
