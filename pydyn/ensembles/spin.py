# https://www.lammps.org/workshops/Aug17/pdf/tranchida.pdf

import cupy as cp
from .base import Ensemble, Operator
from ..constants import Constants


def sib_transform(spins, omegas):
    """
    DOI: 10.1103/PhysRevB.99.224414
    Perform the SIB (Semi-Implicit B) spin update.
    n^p_i = n_i - (n_i + n^p_i) / 2 x (omegas_i)
    """
    A = 0.5 * omegas  # shape (n,3)
    detAi = 1.0 / (1.0 + cp.sum(A**2, axis=1, keepdims=True))  # shape (n, 1)
    a2 = spins - cp.cross(spins, A)  # predictor, shape (n,3)

    #################
    # TODO: which is faster?
    # A_dot_a2 = cp.sum(A * a2, axis=1, keepdims=True)
    # out = a2 + cp.cross(A, a2) + A * A_dot_a2
    # out *= detAi
    #################

    # Precompute repeated terms
    A0, A1, A2 = A[:, 0], A[:, 1], A[:, 2]
    a20, a21, a22 = a2[:, 0], a2[:, 1], a2[:, 2]

    out = cp.zeros_like(spins)
    out[:, 0] = a20 * (A0 * A0 + 1) + a21 * (A0 * A1 - A2) + a22 * (A0 * A2 + A1)
    out[:, 1] = a20 * (A1 * A0 + A2) + a21 * (A1 * A1 + 1) + a22 * (A1 * A2 - A0)
    out[:, 2] = a20 * (A2 * A0 - A1) + a21 * (A2 * A1 + A0) + a22 * (A2 * A2 + 1)
    out *= detAi
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
        self.drift_prefactor = self.drift_base * state.spin.inv_magnitude * dt
        self.noise_prefactor = self.noise_base * cp.sqrt(
            state.spin.inv_magnitude * kT * dt
        )

        noise = cp.random.randn(state.N, 3)
        omega = self.calc_omega(state, context, noise)
        spins_predict = sib_transform(state.spin.vector, omega)
        state_p = state.copy()
        state_p.spin.vector = (spins_predict + state.spin.vector) / 2
        omega = self.calc_omega(state_p, context, noise)
        state.spin.vector = sib_transform(state.spin.vector, omega)

    def calc_omega(self, state, context, noise):

        self.force_model.compute(state, context, properties=["spin_torques"])
        B_eff = self.force_model.results["spin_torques"]
        omega = (
            self.drift_prefactor[:, None]
            * (B_eff + cp.cross(state.spin.vector, B_eff) * self.alpha_t)
            + self.noise_prefactor[:, None] * noise
        )
        return omega


class SpinLLG(Ensemble):

    def __init__(self, force_model, alpha_t=0.1):
        self.force_model = force_model
        self.op_list = [
            (SIBSpinOp(force_model, alpha_t), 1.0),
        ]
