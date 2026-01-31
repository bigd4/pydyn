import cupy as cp
import numpy as np
from .constants import Constants


class Thermal:
    """计算 simulation 中的各种热力学量"""

    def __init__(self, sim):
        self.sim = sim

    @property
    def kinetic_energy(self):
        return self.sim.state.kinetic_energy

    @property
    def temperature(self):
        return self.sim.context.get_temperature(self.sim.state)
