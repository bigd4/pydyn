from .constants import Constants


class SimulationContext:
    def __init__(
        self,
        target_temp=None,
        target_pressure=None,
        constraints=None,
    ):
        self.target_temp = target_temp
        self.target_pressure = target_pressure
        self.constraints = constraints

    def get_temperature(self, state):
        dof = 3 * state.N - sum(c.removed_dof for c in self.constraints)
        return 2 * state.kinetic_energy / (dof * Constants.kB)
