from .base import Ensemble, Operator
import cupy as cp
from ..constants import Constants

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


class MTTKNPTBarostatOp:
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
        if "MTTKBarostat" in state.components:
            return  # already extended
        state.components.append("MTTKBarostat")
        kT = Constants.kB * context.target_temp  # eV
        state.W = (state.N + 1) * kT * self.t_tau**2  # barostat mass
        state.box_p = cp.zeros((3, 3))  # barostat box momenta
        state.R = cp.zeros(self.pchain + 1)  # barostat chain masses
        cell_dof = 9
        state.R[0] = cell_dof * kT * self.t_tau**2  # barostat mass for first chain
        state.R[1:-1] = kT * self.t_tau**2  # barostat mass for other chains
        state.R[-1] = 1.0  # dummy mass for last chain
        state.xi = cp.zeros(self.pchain + 1)  # barostat chain positions
        state.p_xi = cp.zeros(self.pchain + 1)  # barostat chain momenta

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
        state.xi += dt * state.p_xi / state.R

        # ---- scale momenta ----
        state.box_p *= cp.exp(-dt * state.p_xi[0] / state.R[0])

        # ---- forward half ----
        for j in range(self.pchain):
            self._integrate_p_xi_j(state, context, j, dt2, dt4)

    def _integrate_p_xi_j(self, state, context, j, dt2, dt4):
        kT = Constants.kB * context.target_temp  # eV
        state.p_xi[j] *= cp.exp(-dt4 * state.p_xi[j + 1] / state.R[j + 1])
        if j == 0:
            # TODO: do we need to substitute 9 with cell_dof?
            g_j = cp.sum(state.box_p**2) / state.W - 9 * kT
        else:
            g_j = state.p_xi[j - 1] ** 2 / state.R[j - 1] - kT
        state.p_xi[j] += dt2 * g_j
        state.p_xi[j] *= cp.exp(-dt4 * state.p_xi[j + 1] / state.R[j + 1])


class NoseHooverChainThermostatOp:
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
        if "NHThermostat" in state.components:
            return  # already extended
        state.components.append("NHThermostat")
        kT = Constants.kB * context.target_temp

        state.Q = cp.zeros(self.tchain + 1)
        state.Q[0] = 3 * state.N * kT * self.t_tau**2
        state.Q[1:-1] = kT * self.t_tau**2
        state.Q[-1] = 1.0

        state.eta = cp.zeros(self.tchain + 1)
        state.p_eta = cp.zeros(self.tchain + 1)

    def get_thermostat_energy(self, state, context) -> float:
        """Return energy-like contribution from the thermostat variables.
        // thermostat chain energy is equivalent to Eq. (2) in
        // Martyna, Tuckerman, Tobias, Klein, Mol Phys, 87, 1117
        // Sum(0.5*p_eta_k^2/Q_k,k=1,M) + L*k*T*eta_1 + Sum(k*T*eta_k,k=2,M),
        // where L = tdof
        //       M = mtchain
        //       p_eta_k = Q_k*eta_dot[k-1]
        //       Q_1 = L*k*T/t_freq^2
        //       Q_k = k*T/t_freq^2, k > 1
        """
        kT = Constants.kB * context.target_temp
        energy = (
            3 * state.N * kT * state.eta[0]
            + kT * cp.sum(state.eta[1:])
            + cp.sum(0.5 * state.p_eta**2 / state.Q)
        )
        return float(energy)

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
        state.eta += dt * state.p_eta / state.Q

        # ---- scale atomic momenta, and so kinetic energy changed ----
        exp_factor = cp.exp(-dt * state.p_eta[0] / state.Q[0])
        ke_current *= exp_factor**2
        state.p *= exp_factor

        # ---- forward half ----
        for j in range(self.tchain):
            self._integrate_p_eta_j(state, context, ke_current, j, dt2, dt4)

    def _integrate_p_eta_j(self, state, context, ke_current, j, dt2, dt4):
        kT = Constants.kB * context.target_temp
        state.p_eta[j] *= cp.exp(-dt4 * state.p_eta[j + 1] / state.Q[j + 1])
        # j=0处和动量耦合，j>0处和前一级链耦合
        if j == 0:
            # TODO: 3N -> tdof
            g_j = ke_current - 3 * state.N * kT
        else:
            g_j = state.p_eta[j - 1] ** 2 / state.Q[j - 1] - kT  # eV

        state.p_eta[j] += dt2 * g_j  # eV*ps
        state.p_eta[j] *= cp.exp(-dt4 * state.p_eta[j + 1] / state.Q[j + 1])
