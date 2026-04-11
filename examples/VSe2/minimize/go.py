"""Spin relaxation of VSe2 from random magnetic moments using FIRE.
使用 FIRE 从随机磁矩弛豫 VSe2 自旋结构。

VSe2 is a 2D ferromagnet with in-plane magnetic order. V atoms carry
spin magnitude 1, Se atoms are non-magnetic. Starting from random spin
orientations on V sites, the optimizer should recover in-plane
ferromagnetic order.
/ VSe2 是面内铁磁二维材料。V 原子自旋大小为 1，Se 无磁性。
从 V 位随机自旋方向出发，优化器应恢复面内铁磁序。
"""

import numpy as np
import cupy as cp
from ase.io import read

from pydyn.state import State, SpinExtension
from pydyn.context import SimulationContext
from pydyn.forces.hotpp_force import MiaoForceModel
from pydyn.neighbors.cuda_neighbor import CudaNeighborList
from pydyn.minimize import SpinFilter, FIRE, Minimization
from pydyn.observers import AtomsDump


context = SimulationContext(target_temp=0.0, target_pressure=0.0, constraints=[])

# ── 1. Load structure and randomize V spins ──────────────────────────────────
print("=" * 60)
print("Step 1: Load VSe2 and randomize V spin directions")
print("=" * 60)

atoms = read("../VSe2.traj")
symbols = atoms.get_chemical_symbols()
v_mask = np.array([s == 'V' for s in symbols])
N_V = v_mask.sum()

print(f"  N atoms: {len(atoms)} ({N_V} V + {len(atoms)-N_V} Se)")

# Randomize V spin directions on unit sphere, keep Se at zero
rng = np.random.default_rng(42)
spins = np.array(atoms.info['spin'], dtype=np.float64)
random_dirs = rng.standard_normal((N_V, 3))
random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
spins[v_mask] = random_dirs  # magnitude=1, random direction

atoms.info['spin'] = spins
state = State(extensions=[SpinExtension()]).from_atoms(atoms)

# Initial magnetic moment (should be ~0 for random spins)
m_init = cp.asnumpy(state.spin.magnetic_moment)
print(f"  Initial net moment: [{m_init[0]:.4f}, {m_init[1]:.4f}, {m_init[2]:.4f}]")
print(f"  |m|/N_V = {np.linalg.norm(m_init):.4f}  (0 = random, 1 = ferromagnetic)")

# ── 2. FIRE spin relaxation ──────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 2: FIRE spin relaxation (SpinFilter only)")
print("=" * 60)

nb = CudaNeighborList(len(atoms), 60, 6.0)
force_model = MiaoForceModel(nb, "../model.pt", spin=True)

filt = SpinFilter(state)
opt = FIRE(filt, dt_start=0.1, dt_max=1.0)

print(f"  {'step':>6}  {'fmax':>12}  {'|m|/N_V':>10}  {'mz/N_V':>10}")

def log(mini):
    if mini.step_count % 10 == 0 or mini.step_count == 1:
        fmax = float(cp.max(cp.abs(opt._last_forces)))
        m = cp.asnumpy(state.spin.magnetic_moment)
        m_norm = np.linalg.norm(m)
        print(f"  {mini.step_count:6d}  {fmax:12.5f}  {m_norm:10.4f}  {m[2]:10.4f}")

driver = Minimization(
    opt, force_model, context,
    observers=[log, AtomsDump("trajectory.xyz", interval=5)],
)
n = driver.run(max_steps=100, fmax=0.01)
print(f"\n  Done in {n} steps.")

# ── 3. Verify: in-plane ferromagnetic order? ─────────────────────────────────
print()
print("=" * 60)
print("Step 3: Verify magnetic order")
print("=" * 60)

m_final = cp.asnumpy(state.spin.magnetic_moment)
m_norm = np.linalg.norm(m_final)
m_xy = np.linalg.norm(m_final[:2])

# Per-atom spin alignment: dot product of each V spin with mean direction
v_spins = cp.asnumpy(state.spin.vector)[v_mask]
mean_dir = v_spins.mean(axis=0)
mean_dir /= np.linalg.norm(mean_dir)
alignment = v_spins @ mean_dir  # 1 = aligned, 0 = random

print(f"  Net moment   : [{m_final[0]:.4f}, {m_final[1]:.4f}, {m_final[2]:.4f}]")
print(f"  |m|/N_V      : {m_norm:.4f}  (1.0 = perfect ferromagnet)")
print(f"  |m_xy|/N_V   : {m_xy:.4f}  (in-plane component)")
print(f"  |m_z|/N_V    : {abs(m_final[2]):.4f}  (out-of-plane component)")
print(f"  Mean alignment: {alignment.mean():.4f}  (1.0 = all parallel)")

is_inplane_ferro = (
    m_norm > 0.9
    and m_xy / max(m_norm, 1e-10) > 0.9
    and alignment.mean() > 0.9
)
print(f"\n  In-plane ferromagnetic: {is_inplane_ferro}")
