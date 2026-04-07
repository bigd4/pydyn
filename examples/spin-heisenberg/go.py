"""
30x30x30 cubic Heisenberg model with LLG dynamics at 20K.
30x30x30 立方海森堡模型在 20K 下的 LLG 动力学模拟。

This example simulates a 3D cubic lattice of classical spins using the
Heisenberg exchange model with Landau-Lifshitz-Gilbert (LLG) dynamics.

该示例使用海森堡交换模型和朗道-利夫希茨-吉尔伯特（LLG）动力学模拟
经典自旋的三维立方晶格。

System parameters / 系统参数:
    - Lattice: 30x30x30 simple cubic lattice (27,000 spins)
    - Exchange coupling J = 1.0 eV (ferromagnetic)
    - Temperature: 20 K
    - Damping parameter α = 0.1
    - Time step: 1 fs
"""

import sys
import numpy as np
import cupy as cp

# Add pydyn to path if needed
sys.path.insert(0, "/home/gegejun/src/PyDyn")

from pydyn.state import State, SpinExtension
from pydyn.context import SimulationContext
from pydyn.ensembles.spin import SpinLLG
from pydyn.simulation import Simulation
from pydyn.forces.heisenberg_force import HeisenbergForceModel
from pydyn.observers import LogThermol
from pydyn.constants import Constants


# Simulation parameters / 模拟参数
L = 30  # Lattice size / 晶格尺寸 (30x30x30)
T = 20.0  # Temperature in K / 温度（K）
dt = 0.001  # Time step in ps / 时间步长（ps）
n_steps = 10000  # Number of steps / 步数

# Model parameters / 模型参数
J = 1.0  # Exchange coupling in eV (positive = ferromagnetic) / 交换耦合（eV）
a = 2.5  # Lattice constant in Angstroms / 晶格常数（埃）
cutoff = a * 1.1  # Neighbor cutoff (slightly larger than a) / 近邻截断
alpha_t = 0.1  # LLG damping parameter / LLG 阻尼参数

print(f"Setting up {L}x{L}x{L} cubic Heisenberg model...")
print(f"Total spins: {L**3}")
print(f"Temperature: {T} K")
print(f"Exchange coupling J: {J} eV")
print(f"LLG damping α: {alpha_t}")

# Generate cubic lattice positions / 生成立方晶格位置
N = L**3
positions = np.zeros((N, 3), dtype=np.float64)
spins = np.zeros((N, 3), dtype=np.float64)

idx = 0
for ix in range(L):
    for iy in range(L):
        for iz in range(L):
            positions[idx] = [ix * a, iy * a, iz * a]
            # Initialize with ferromagnetic alignment along z / 沿 z 方向铁磁排列初始化
            spins[idx] = [0.0, 0.0, 1.0]
            idx += 1

# Create simulation box / 创建模拟盒
box = np.array([[L * a, 0, 0], [0, L * a, 0], [0, 0, L * a]], dtype=np.float64)

# Create state with spin extension / 创建带有自旋扩展的状态
r = cp.array(positions)
p = cp.zeros((N, 3), dtype=cp.float64)
m = cp.ones(N, dtype=cp.float64)  # Dummy masses / 虚拟质量
atomic_number = cp.ones(N, dtype=cp.int32) * 26  # Fe / 铁
pbc = cp.array([True, True, True])
spin_ext = SpinExtension(cp.array(spins))

state = State(
    r=r, p=p, m=m, box=box, atomic_number=atomic_number, pbc=pbc, extensions=[spin_ext]
)

print(f"System volume: {float(state.volume):.2f} Å³")

# Create simulation context / 创建模拟环境
context = SimulationContext(target_temp=T, target_pressure=None, constraints=[])

# Create Heisenberg force model / 创建海森堡力模型
force_model = HeisenbergForceModel(J=J, cutoff=cutoff)

# Create LLG ensemble / 创建 LLG 系综
ensemble = SpinLLG(force_model=force_model, alpha_t=alpha_t)


# Set up observers / 设置观测器
def get_magnetization(sim):
    """Calculate total magnetization. / 计算总磁化强度。"""
    spin_vector = sim.state.spin.vector
    total_spin = cp.sum(spin_vector, axis=0)
    magnitude = cp.linalg.norm(total_spin) / sim.state.N
    return float(magnitude)


def get_spin_temperature(sim):
    """Calculate effective spin temperature. / 计算有效自旋温度。"""
    return sim.state.spin.get_spin_temperature(
        sim.state, sim.context, sim.ensemble.force_model
    )


log_observer = LogThermol(
    "log.txt",
    {
        "Step": lambda sim: sim.step,
        "Magnetization": get_magnetization,
        "SpinTemp": get_spin_temperature,
    },
    interval=100,
)

# Create and run simulation / 创建并运行模拟
print("\nStarting simulation...")
sim = Simulation(
    state=state,
    ensemble=ensemble,
    context=context,
    observers=[log_observer],
    dt=dt,
)

sim.run(n_steps)

print(f"\nSimulation completed after {n_steps} steps.")
print(f"Final magnetization: {get_magnetization(sim):.6f}")
