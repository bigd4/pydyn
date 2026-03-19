import cupy as cp
from .constants import Constants
from ase import Atoms


class State:
    def __init__(
        self,
        r=None,
        p=None,
        m=None,
        box=None,
        atomic_number=None,
        pbc=None,
        extensions=None,
    ):
        self.N = len(r) if r is not None else 0
        self.r = r
        self.p = p
        self.m = m
        self.box = box
        self.atomic_number = atomic_number
        self.pbc = pbc
        self.components = {}
        if extensions is not None:
            for ext in extensions:
                self.add_component(ext)

    def add_component(self, component, descriptor=None):
        if descriptor is None:
            descriptor = component.descriptor
        self.components[descriptor] = component
        setattr(self, descriptor, component)

    def remove_component(self, descriptor):
        if descriptor in self.components:
            del self.components[descriptor]
            delattr(self, descriptor)

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
        for ext in self.components.values():
            ext.from_atoms(atoms)
        return self

    def to_atoms(self):
        atoms = Atoms(
            symbols=self.atomic_number,
            positions=cp.asnumpy(self.r),
            masses=cp.asnumpy(self.m),
            cell=cp.asnumpy(self.box),
            pbc=self.pbc,
        )
        for ext in self.components.values():
            ext.to_atoms(atoms)
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
        for ext_name, ext in self.components.items():
            if ext_name not in state2.components:
                return False
            if not ext.configure_same_as(state2.components[ext_name]):
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
        for ext_name, ext in self.components.items():
            new_state.add_component(ext.copy(), descriptor=ext_name)
        return new_state


class Extension:
    descriptor = None

    def from_atoms(self, atoms): ...
    def to_atoms(self, atoms): ...
    def copy(self): ...
    def configure_same_as(self, other): ...


class SpinExtension(Extension):

    descriptor = "spin"

    def __init__(self, spins=None):
        self.magnitude = None
        self.vector = None
        self.inv_magnitude = None
        self.spins = spins

    @property
    def spins(self):
        if self.vector is None or self.magnitude is None:
            return None
        return self.vector * self.magnitude[:, None]

    @spins.setter
    def spins(self, spins):
        if spins is not None:
            self.magnitude = cp.linalg.norm(spins, axis=1)
            self.inv_magnitude = cp.where(
                self.magnitude > 0.01, 1.0 / self.magnitude, 0.0
            )
            self.vector = spins * self.inv_magnitude[:, None]

    @property
    def magnetic_moment(self):
        if self.spins is None:
            return None
        return cp.sum(self.spins, axis=0) / cp.sum(self.magnitude)

    def from_atoms(self, atoms):
        self.spins = cp.array(atoms.info["spin"])

    def to_atoms(self, atoms):
        atoms.info["spin"] = cp.asnumpy(self.spins)

    def configure_same_as(self, other):
        return cp.allclose(self.spins, other.spins, atol=1e-7)

    def copy(self):
        new = self.__class__(spins=self.spins)
        return new

    def get_spin_temperature(self, state, context, force_model):
        force_model.compute(state, context, properties=["spin_torques"])
        B_eff = force_model.results["spin_torques"]
        spin_temp = (
            cp.sum(cp.linalg.norm(cp.cross(self.spins, B_eff), axis=1) ** 2)
            / cp.sum(self.spins * B_eff)
            / (2 * Constants.kB)
        )
        return spin_temp


class MTTKBarostatExtension(Extension):

    descriptor = "mttk_barostat"

    def __init__(self, W=None, box_p=None, R=None, xi=None, p_xi=None):
        self.W = W
        self.box_p = box_p
        self.R = R
        self.xi = xi
        self.p_xi = p_xi

    def copy(self):
        new = self.__class__(
            W=self.W.copy() if self.W is not None else None,
            box_p=self.box_p.copy() if self.box_p is not None else None,
            R=self.R.copy() if self.R is not None else None,
            xi=self.xi.copy() if self.xi is not None else None,
            p_xi=self.p_xi.copy() if self.p_xi is not None else None,
        )
        return new

    def configure_same_as(self, other):
        return True  # 一般 barostat 不需要严格匹配


class NHThermostatExtension(Extension):

    descriptor = "nh_thermostat"

    def __init__(self, Q=None, eta=None, p_eta=None):
        self.Q = Q
        self.eta = eta
        self.p_eta = p_eta

    def copy(self):
        new = self.__class__(
            Q=self.Q.copy() if self.Q is not None else None,
            eta=self.eta.copy() if self.eta is not None else None,
            p_eta=self.p_eta.copy() if self.p_eta is not None else None,
        )
        return new

    def configure_same_as(self, other):
        return True  # 一般 thermostat 不需要严格匹配

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
            3 * state.N * kT * self.eta[0]
            + kT * cp.sum(self.eta[1:])
            + cp.sum(0.5 * self.p_eta**2 / self.Q)
        )
        return float(energy)
