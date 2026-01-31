from ase.calculators.emt import EMT
from .base import ForceModel
from ase import Atoms
import cupy as cp


class EMTForceModel(ForceModel):
    def __init__(self):
        super().__init__()
        self.calculator = EMT()

    def compute(self, state, context):
        super().compute(state, context)
        atoms = state.to_atoms()
        atoms.calc = self.calculator
        self.results["potential_energy"] = cp.array(atoms.get_potential_energy())
        self.results["forces"] = cp.array(atoms.get_forces())
        self.results["virial"] = (
            atoms.get_volume() * cp.array(atoms.get_stress(voigt=False)) * -1.0
        )
