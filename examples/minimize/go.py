"""Variable-cell structural optimization of 8-atom diamond using FIRE.
使用 FIRE 对 8 原子金刚石结构进行变胞结构优化。

Starts from a randomly perturbed diamond unit cell, relaxes both atomic
positions and cell vectors simultaneously, then verifies the result is
diamond.
"""

import numpy as np
import cupy as cp
from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list

from pydyn.state import State
from pydyn.context import SimulationContext
from pydyn.forces.hotpp_force import MiaoForceModel
from pydyn.neighbors.small_cell_neighbor import SmallCellNeighborList
from pydyn.minimize import AtomFilter, CellFilter, CompositeFilter, FIRE, Minimization
from pydyn.observers import AtomsDump


context = SimulationContext(target_temp=0.0, target_pressure=0.0, constraints=[])

# ── 1. Build reference diamond ──────────────────────────────────────────────
print("=" * 60)
print("Step 1: Reference diamond structure")
print("=" * 60)

ref = bulk('C', 'diamond', a=3.567, cubic=True)
state_ref = State().from_atoms(ref)
nb = SmallCellNeighborList(cutoff=3.0)
force_model = MiaoForceModel(nb, "model.pt")
force_model.compute(state_ref, context, properties=["forces", "energy"])
f_ref = float(cp.max(cp.abs(force_model.results["forces"])))

print(f"  N atoms: {len(ref)}")
print(f"  Cell   : {ref.cell[0,0]:.4f} Ang (cubic)")
print(f"  Volume : {ref.get_volume():.4f} Ang^3")
print(f"  fmax   : {f_ref:.6f} eV/Ang (near zero = local minimum)")

# ── 2. Apply random perturbation ─────────────────────────────────────────────
print()
print("=" * 60)
print("Step 2: Apply random perturbation")
print("=" * 60)

rng = np.random.default_rng(42)
atoms = ref.copy()
atoms.positions += rng.uniform(-0.25, 0.25, size=atoms.positions.shape)
strain = np.eye(3) + rng.uniform(-0.06, 0.06, size=(3, 3))
atoms.cell[:] = atoms.cell @ strain
atoms.wrap()

vol_ref = ref.get_volume()
vol_pert = atoms.get_volume()
print(f"  Position noise: ±0.25 Ang")
print(f"  Cell strain   : ±6%")
print(f"  Volume (ref)  : {vol_ref:.4f} Ang^3")
print(f"  Volume (pert) : {vol_pert:.4f} Ang^3")
# ── 3. FIRE minimization ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 3: FIRE minimization (atoms + cell)")
print("=" * 60)

state = State().from_atoms(atoms)
nb = SmallCellNeighborList(cutoff=3.0)
force_model = MiaoForceModel(nb, "model.pt")

filt = CompositeFilter(state, [AtomFilter(state), CellFilter(state)])
opt = FIRE(filt, dt_start=0.005, dt_max=0.05)

print(f"  {'step':>6}  {'fmax (eV/Ang)':>14}  {'volume (Ang^3)':>14}")

def log(mini):
    if mini.step_count % 20 == 0 or mini.step_count == 1:
        fmax = float(cp.max(cp.abs(opt._last_forces)))
        vol = float(state.volume)
        print(f"  {mini.step_count:6d}  {fmax:14.5f}  {vol:14.4f}")

driver = Minimization(
    opt, force_model, context,
    observers=[log, AtomsDump("trajectory.xyz", interval=10)],
)
n = driver.run(max_steps=2000, fmax=0.01)
print(f"\n  Done in {n} steps.")

# ── 4. Verify: is it diamond? ────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 4: Verify relaxed structure")
print("=" * 60)

result = state.to_atoms()
write("relaxed.xyz", result)

vol_final = result.get_volume()
print(f"  Volume (relaxed) : {vol_final:.4f} Ang^3")
print(f"  Volume (ref)     : {vol_ref:.4f} Ang^3")
print(f"  Volume error     : {abs(vol_final - vol_ref)/vol_ref*100:.2f}%")

# Bond lengths and coordination (diamond: 4 nearest neighbors at ~1.54 Ang)
i_idx, j_idx, d = neighbor_list('ijd', result, cutoff=2.0)
cn = np.bincount(i_idx, minlength=len(result))

print(f"\n  Bond lengths (cutoff 2.0 Ang):")
print(f"    mean = {d.mean():.4f} Ang  (diamond: 1.5446 Ang)")
print(f"    std  = {d.std():.4f} Ang")
print(f"    min  = {d.min():.4f} Ang")
print(f"    max  = {d.max():.4f} Ang")
print(f"\n  Coordination numbers: {cn.tolist()}  (all 4 = diamond)")

is_diamond = (
    bool(np.all(cn == 4))
    and abs(vol_final - vol_ref) / vol_ref < 0.05
    and 1.50 < d.mean() < 1.60
)
print(f"\n  Is diamond: {is_diamond}")
