from .base import Ensemble, Operator
import cupy as cp
from ..constants import Constants
from ..state import MTTKBarostatExtension, NHThermostatExtension

"""
nose hoove chain:
p_chain[j] ← exp(−ξ[j+1]) · (p_chain[j] + Δt·g_j) · exp(−ξ[j+1])
Thermostat	          Barostat
eta, p_eta, Q	      xi, p_xi, R
p                     box_p
sum(p^2/m)	          sum(box_p^2)/W  lammps use different W
dof * kT (3NkT)	      cell_dof * kT (9kT)
"""

FOURTH_ORDER_COEFFS = cp.array(
    [
        1 / (2 - 2 ** (1 / 3)),
        -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
        1 / (2 - 2 ** (1 / 3)),
    ]
)


class MTTKNPTBarostatOp(Operator):
    """
    MTTK NPT Barostat Operator (full anisotropic, multi-chain)
    Updates atomic momenta and box according to target pressure.
    """

    def __init__(self, t_tau, pchain=3, ploop=1):
        self.t_tau = t_tau  # barostat relaxation time
        self.pchain = pchain  # Nosé-Hoover chain length
        self.ploop = ploop  # number of subloops for 4th-order integrator

    def extend_state(self, state, context):
        """
        state: requires
            - p: atomic momenta, (N,3)                         [amu*A/ps]
            - box: cell vectors, (3,3)                         [A]
            ------------------------------- extend:
            - xi: barostat chain positions, (pchain,)          [unitless]
            - p_xi: barostat chain momenta, (pchain,)          [eV*ps]
            - R: barostat chain masses, (pchain,)              [eV*ps^2]
            - box_p: barostat box momenta, (3,3)               [eV*ps]
            - W: barostat mass                                 [eV*ps^2]
            -------------------------------------------
        context: provides
            - target_pressure
            - kB
        """
        if "mttk_barostat" in state.components:
            return  # already extended

        kT = Constants.kB * context.target_temp  # eV
        W = (state.N + 1) * kT * self.t_tau**2  # barostat mass
        box_p = cp.zeros((3, 3))  # barostat box momenta
        R = cp.zeros(self.pchain + 1)  # barostat chain masses
        cell_dof = 9
        R[0] = cell_dof * kT * self.t_tau**2  # barostat mass for first chain
        R[1:-1] = kT * self.t_tau**2  # barostat mass for other chains
        R[-1] = 1.0  # dummy mass for last chain
        xi = cp.zeros(self.pchain + 1)  # barostat chain positions
        p_xi = cp.zeros(self.pchain + 1)  # barostat chain momenta

        barostat = MTTKBarostatExtension(W=W, box_p=box_p, R=R, xi=xi, p_xi=p_xi)
        state.add_component(barostat)

    def apply(self, state, context, dt):
        # 4th-order loop
        for _ in range(self.ploop):
            for coeff in FOURTH_ORDER_COEFFS:
                dt_sub = coeff * dt / self.ploop
                self._integrate_step(state, context, dt_sub)

    def _integrate_step(self, state, context, dt):
        dt2 = dt / 2
        dt4 = dt / 4

        # ---- backward half (reverse order) ----
        for j in reversed(range(self.pchain)):
            self._integrate_p_xi_j(state, context, j, dt2, dt4)

        # ---- update xi ----
        state.mttk_barostat.xi += dt * state.mttk_barostat.p_xi / state.mttk_barostat.R

        # ---- scale momenta ----
        state.mttk_barostat.box_p *= cp.exp(
            -dt * state.mttk_barostat.p_xi[0] / state.mttk_barostat.R[0]
        )

        # ---- forward half ----
        for j in range(self.pchain):
            self._integrate_p_xi_j(state, context, j, dt2, dt4)

    def _integrate_p_xi_j(self, state, context, j, dt2, dt4):
        kT = Constants.kB * context.target_temp  # eV
        state.mttk_barostat.p_xi[j] *= cp.exp(
            -dt4 * state.mttk_barostat.p_xi[j + 1] / state.mttk_barostat.R[j + 1]
        )
        if j == 0:
            # TODO: do we need to substitute 9 with cell_dof?
            g_j = cp.sum(state.mttk_barostat.box_p**2) / state.mttk_barostat.W - 9 * kT
        else:
            g_j = (
                state.mttk_barostat.p_xi[j - 1] ** 2 / state.mttk_barostat.R[j - 1] - kT
            )
        state.mttk_barostat.p_xi[j] += dt2 * g_j
        state.mttk_barostat.p_xi[j] *= cp.exp(
            -dt4 * state.mttk_barostat.p_xi[j + 1] / state.mttk_barostat.R[j + 1]
        )


class NoseHooverChainThermostatOp(Operator):
    """
    Nose-Hoover Chain Thermostat Operator (NVT part)
    """

    def __init__(self, t_tau, tchain=3, tloop=1):
        self.t_tau = t_tau
        self.tchain = tchain
        self.tloop = tloop

    def extend_state(self, state, context):
        """
        Extend state with thermostat variables

        requires:
            - p: atomic momenta, (N,3)
            - masses: atomic masses, (N,)

        extends:
            - eta: thermostat chain positions, (tchain,)         # [unitless]
            - p_eta: thermostat chain momenta, (tchain,)         # [eV*ps]
            - Q: thermostat chain masses, (tchain,)              # [eV*ps^2]

        """

        if "nh_thermostat" in state.components:
            return  # already extended
        kT = Constants.kB * context.target_temp

        Q = cp.zeros(self.tchain + 1)
        Q[0] = 3 * state.N * kT * self.t_tau**2
        Q[1:-1] = kT * self.t_tau**2
        Q[-1] = 1.0

        eta = cp.zeros(self.tchain + 1)
        p_eta = cp.zeros(self.tchain + 1)

        thermostat = NHThermostatExtension(Q=Q, eta=eta, p_eta=p_eta)
        state.add_component(thermostat)

    def apply(self, state, context, dt):
        for _ in range(self.tloop):
            for coeff in FOURTH_ORDER_COEFFS:
                dt_sub = coeff * dt / self.tloop
                self._integrate_step(state, context, dt_sub)

    def _integrate_step(self, state, context, dt):
        dt2 = dt / 2
        dt4 = dt / 4
        ke_current = cp.sum(state.p**2 / state.m[:, None]) * Constants.mv2_to_e  # eV
        # ---- backward half ----
        for j in reversed(range(self.tchain)):
            self._integrate_p_eta_j(state, context, ke_current, j, dt2, dt4)

        # ---- update eta ----
        state.thermostat.eta += dt * state.thermostat.p_eta / state.thermostat.Q

        # ---- scale atomic momenta, and so kinetic energy changed ----
        exp_factor = cp.exp(-dt * state.thermostat.p_eta[0] / state.thermostat.Q[0])
        ke_current *= exp_factor**2
        state.p *= exp_factor

        # ---- forward half ----
        for j in range(self.tchain):
            self._integrate_p_eta_j(state, context, ke_current, j, dt2, dt4)

    def _integrate_p_eta_j(self, state, context, ke_current, j, dt2, dt4):
        kT = Constants.kB * context.target_temp
        state.thermostat.p_eta[j] *= cp.exp(
            -dt4 * state.thermostat.p_eta[j + 1] / state.thermostat.Q[j + 1]
        )
        # j=0处和动量耦合，j>0处和前一级链耦合
        if j == 0:
            # TODO: 3N -> tdof
            g_j = ke_current - 3 * state.N * kT
        else:
            g_j = (
                state.thermostat.p_eta[j - 1] ** 2 / state.thermostat.Q[j - 1] - kT
            )  # eV

        state.thermostat.p_eta[j] += dt2 * g_j  # eV*ps
        state.thermostat.p_eta[j] *= cp.exp(
            -dt4 * state.thermostat.p_eta[j + 1] / state.thermostat.Q[j + 1]
        )
