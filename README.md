# PyDyn - GPU-Accelerated Molecular Dynamics

PyDyn is a GPU-accelerated molecular dynamics simulation framework leveraging CUDA and CuPy for high-performance neighbor search and force calculations.

## Prerequisites

- Python 3.8 or higher
- CUDA Toolkit 11.0 or later
- NVIDIA GPU with compute capability 8.0 or higher (Ampere architecture and newer)
- CMake 3.15 or later

## Dependencies

The project requires the following Python packages:

- `ase>=3.20` - Atomic simulation environment
- `cupy>=11.0` - GPU array computing library
- `torch>=1.9` - PyTorch deep learning framework
- `scipy>=1.5` - Scientific computing utilities

## Installation

Clone the repository with its submodules:

```bash
git clone --recursive git@gitlab.com:bigd4/pydyn.git
cd pydyn
```

Install in development mode:

```bash
pip install -e .
```

## Quick Start

Create a simulation with the NVT Nose-Hoover ensemble:

```python
from pydyn.state import State
from pydyn.context import SimulationContext
from pydyn.ensembles.nvt import NVTNoseHoover
from pydyn.simulation import Simulation
from pydyn.initializer import MaxwellBoltzmannDistribution
from pydyn.constraints import RemoveCOMMomentum
from pydyn.forces.hotpp_force import MiaoForceModel
from pydyn.neighbors.cuda_neighbor import CudaNeighborList
from ase.io import read

# Load atomic structure
atoms = read("structure.xyz")
state = State().from_atoms(atoms)

# Setup simulation context
context = SimulationContext(
    target_temp=300,
    constraints=[RemoveCOMMomentum()],
)

# Create neighbor list and force model
nb = CudaNeighborList(len(atoms), 30, 3.0)
force_model = MiaoForceModel(nb, "model.pt")

# Create ensemble and run simulation
ensemble = NVTNoseHoover(t_tau=0.1, force_model=force_model)
sim = Simulation(
    state=state,
    ensemble=ensemble,
    context=context,
    initializer=[MaxwellBoltzmannDistribution(target_temp=300)],
    dt=0.001,
)

sim.run(10000)
```

See `examples/nvt-nhc/go.py` for a complete working example.

