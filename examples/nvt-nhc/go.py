import sys
import pydyn
from pydyn.state import State
from pydyn.context import SimulationContext
from pydyn.ensembles.nvt import NVTNoseHoover
from pydyn.simulation import Simulation
from datetime import datetime
from ase.lattice.cubic import FaceCenteredCubic as LatticeFCC
from pydyn.initializer import MaxwellBoltzmannDistribution
from pydyn.constraints import RemoveCOMMomentum
from pydyn.forces.hotpp_force import MiaoForceModel
from pydyn.neighbors.cuda_neighbor import CudaNeighborList
from pydyn.observers import LogThermol, AtomsDump
from pydyn.constants import Constants
import cupy as cp
from ase.io import read

T = 300
dt = 0.001

atoms = read("carbon.xyz")
print(atoms)
state = State().from_atoms(atoms)

context = SimulationContext(
    target_temp=T,
    target_pressure=0.0,
    constraints=[RemoveCOMMomentum()],
)

nb = CudaNeighborList(len(atoms), 30, 3.0)
force_model = MiaoForceModel(nb, "model.pt")

# ensemble = MTTKNPT(t_tau=100 * dt, p_tau=1000 * dt, force_model=force_model)
ensemble = NVTNoseHoover(t_tau=100 * dt, force_model=force_model)

log_observer = LogThermol(
    "log.txt",
    {
        "Temp": lambda sim: sim.context.get_temperature(sim.state),
        "conserveE": lambda sim: sim.ensemble.get_conserved_energy(
            sim.state, sim.context
        ),
        # "pressure": lambda sim: sim.ensemble.get_pressure(sim.state, sim.context),
    },
    interval=100,
)
atoms_dump = AtomsDump("trajectory.xyz", interval=1000)


sim = Simulation(
    state=state,
    ensemble=ensemble,
    context=context,
    initializer=[MaxwellBoltzmannDistribution(target_temp=T)],
    observers=[log_observer, atoms_dump],
    # observers=[log_observer],
    dt=dt,
)

sim.run(10000)
