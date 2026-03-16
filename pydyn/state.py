import cupy as cp
from .constants import Constants
from ase import Atoms


class State:
    def __init__(self, r=None, p=None, m=None, box=None, atomic_number=None, pbc=None):
        self.N = len(r) if r is not None else 0
        self.r = r
        self.p = p
        self.m = m
        self.box = box
        self.atomic_number = atomic_number
        self.pbc = pbc
        self.components = []

    @property
    def kinetic_energy(self):
        if self.p is None or self.m is None:
            return 0.0
        return 0.5 * cp.sum(self.p**2 / self.m[:, None]) * Constants.mv2_to_e

    @property
    def kinetic_virial(self):
        if self.p is None or self.m is None:
            return cp.zeros((3, 3))
        return self.p.T @ (self.p / self.m[:, None]) * Constants.mv2_to_e  # eV

    @property
    def volume(self):
        if self.box is None:
            return 0.0
        return cp.linalg.det(self.box)

    def from_atoms(self, atoms):
        self.N = len(atoms)
        self.r = cp.array(atoms.get_positions())
        self.p = cp.array(atoms.get_momenta())
        self.m = cp.array(atoms.get_masses())
        self.box = cp.array(atoms.get_cell())
        self.atomic_number = cp.array(atoms.get_atomic_numbers())
        self.pbc = atoms.get_pbc()
        return self

    def to_atoms(self):
        atoms = Atoms(
            symbols=self.atomic_number,
            positions=cp.asnumpy(self.r),
            masses=cp.asnumpy(self.m),
            cell=cp.asnumpy(self.box),
            pbc=self.pbc,
        )
        return atoms

    def configure_same_as(self, state2):
        if state2 is None:
            return False
        if self.N != state2.N:
            return False
        if not cp.allclose(self.r, state2.r, atol=1e-7):
            return False
        if not cp.allclose(self.box, state2.box, atol=1e-7):
            return False
        return True

    def copy(self):
        new_state = self.__class__(
            r=self.r.copy() if self.r is not None else None,
            p=self.p.copy() if self.p is not None else None,
            m=self.m.copy() if self.m is not None else None,
            box=self.box.copy() if self.box is not None else None,
            atomic_number=(
                self.atomic_number.copy() if self.atomic_number is not None else None
            ),
            pbc=self.pbc.copy() if self.pbc is not None else None,
        )
        return new_state


class SpinState(State):
    def __init__(
        self, r=None, p=None, m=None, box=None, atomic_number=None, pbc=None, spins=None
    ):
        super().__init__(r, p, m, box, atomic_number, pbc)
        self.spins = spins

    @property
    def mu_i(self):
        mu_i = cp.linalg.norm(self.spins, axis=1)
        return mu_i

    def from_atoms(self, atoms):
        super().from_atoms(atoms)
        self.spins = cp.array(
            atoms.info["spins"]
        )  # assuming initial magnetic moments are provided
        return self

    def to_atoms(self):
        atoms = super().to_atoms()
        atoms.info["spins"] = cp.asnumpy(self.spins)
        return atoms

    def configure_same_as(self, state2):
        if not super().configure_same_as(state2):
            return False
        return cp.allclose(self.spins, state2.spins, atol=1e-7)

    def copy(self):
        new_state = super().copy()
        new_state.spins = self.spins.copy() if self.spins is not None else None
        return new_state
