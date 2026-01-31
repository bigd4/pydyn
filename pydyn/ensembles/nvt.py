from .base import Ensemble, MomentumOp, PositionOp, Operator
import cupy as cp
from .nhc import NoseHooverChainThermostatOp
from ..constants import Constants


class BerendsenTemperatureOp(Operator):
    def __init__(self, t_tau):
        self.t_tau = t_tau

    def apply(self, state, context, dt):
        temperature = context.get_temperature(state)
        scale = cp.sqrt(1 + dt / self.t_tau * (context.target_temp / temperature - 1))
        scale = cp.clip(scale, 0.9, 1.1)
        state.p *= scale


class NVTBerendsen(Ensemble):

    def __init__(self, t_tau, force_model):
        self.op_list = [
            (BerendsenTemperatureOp(t_tau), 1.0),
            (MomentumOp(force_model), 0.5),
            (PositionOp(), 1.0),
            (MomentumOp(force_model), 0.5),
        ]


class NVTNoseHoover(Ensemble):

    def __init__(self, t_tau, force_model):
        self.t_tau = t_tau
        self.force_model = force_model
        self.op_list = [
            (NoseHooverChainThermostatOp(t_tau), 0.5),
            (MomentumOp(force_model), 0.5),
            (PositionOp(), 1.0),
            (MomentumOp(force_model), 0.5),
            (NoseHooverChainThermostatOp(t_tau), 0.5),
        ]

    def get_conserved_energy(self, state, context):

        kT = Constants.kB * context.target_temp  # eV
        self.force_model.compute(state, context)
        potential_energy = self.force_model.results["potential_energy"]
        thermostat_energy = (
            3 * state.N * kT * state.eta[0]
            + kT * cp.sum(state.eta[1:])
            + cp.sum(0.5 * state.p_eta**2 / state.Q)
        )
        conserved_energy = potential_energy + state.kinetic_energy + thermostat_energy
        return float(conserved_energy)
