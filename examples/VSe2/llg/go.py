import sys
import pydyn
from pydyn.state import State, SpinExtension
from pydyn.context import SimulationContext
from pydyn.ensembles.spin import SpinLLG
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

T = float(sys.argv[1])
dt = 0.001

atoms = read("../VSe2.traj")
print(atoms)
state = State(extensions=[SpinExtension()]).from_atoms(atoms)

context = SimulationContext(
    target_temp=T,
    target_pressure=0.0,
    constraints=[RemoveCOMMomentum()],
)

nb = CudaNeighborList(len(atoms), 60, 6.0)
force_model = MiaoForceModel(nb, "../model.pt", spin=True)

ensemble = SpinLLG(force_model=force_model)

def mu_B(state):
    n = cp.linalg.norm(state.spin.magnetic_moment)
    return float(n)

log_observer = LogThermol(
    "log.txt",
    {
        "targetTemp": lambda sim: sim.context.target_temp,
        "SpinTemp": lambda sim: sim.state.spin.get_spin_temperature(sim.state, sim.context, sim.ensemble.force_model),
        "mu_B": lambda sim: mu_B(sim.state)
    },
    interval=100,
)
atoms_dump = AtomsDump("trajectory.xyz", interval=100)


sim = Simulation(
    state=state,
    ensemble=ensemble,
    context=context,
    initializer=[MaxwellBoltzmannDistribution(target_temp=T)],
    observers=[log_observer, atoms_dump],
    #observers=[log_observer],
    dt=dt,
)

sim.run(50000)

sim.finalize()
