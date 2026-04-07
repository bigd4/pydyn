"""Shared pytest fixtures for PyDyn tests.
PyDyn 测试的共享 pytest fixture。

This module provides reusable fixtures for creating mock states, contexts,
and handling CuPy/NumPy compatibility across tests.
"""

import pytest
import sys
import numpy as np

# Try to import cupy; fallback to numpy if unavailable
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    # Inject numpy as cupy into sys.modules so pydyn modules can use it
    # 将 numpy 作为 cupy 注入到 sys.modules 中，以便 pydyn 模块可以使用它
    sys.modules['cupy'] = np
    cp = np
    HAS_CUPY = False

# Create mock torch and submodules to avoid import errors
# 创建模拟 torch 和子模块以避免导入错误
class MockModule:
    """Mock module that returns itself for any attribute access."""
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter([self])

mock_torch = MockModule()
sys.modules['torch'] = mock_torch
sys.modules['torch.utils'] = MockModule()
sys.modules['torch.utils.dlpack'] = MockModule()

from pydyn.state import State
from pydyn.context import SimulationContext
from pydyn.constants import Constants


@pytest.fixture
def use_numpy():
    """Fixture to skip tests if cupy is required but unavailable.
    如果需要 cupy 但不可用，则跳过测试的 fixture。
    """
    if not HAS_CUPY:
        pytest.skip("CuPy not available, using NumPy arrays")
    return HAS_CUPY


@pytest.fixture
def mock_state():
    """Create a simple mock State object with 3 atoms.
    创建一个包含 3 个原子的简单模拟 State 对象。

    Returns:
        State: A state with known positions, momenta, and masses for testing.
            具有已知位置、动量和质量的状态，用于测试。
    """
    # Use numpy as fallback if cupy unavailable
    array_module = cp if HAS_CUPY else np

    # Create 3 atoms with known initial conditions
    # 创建 3 个原子的已知初始条件
    N = 3
    r = array_module.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=float)

    # Initialize with zero momentum
    # 用零动量初始化
    p = array_module.zeros((N, 3), dtype=float)

    # Use identical masses (1 amu = 1 atomic mass unit)
    # 使用相同的质量（1 amu = 1 原子质量单位）
    m = array_module.ones(N, dtype=float)

    # Simple cubic cell
    # 简单立方盒
    box = array_module.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ], dtype=float)

    # Dummy atomic numbers (e.g., Argon)
    # 虚拟原子序数（例如，氩）
    atomic_number = array_module.array([18, 18, 18], dtype=int)

    # Periodic boundary conditions in all directions
    # 所有方向上的周期性边界条件
    pbc = array_module.array([True, True, True])

    state = State(
        r=r,
        p=p,
        m=m,
        box=box,
        atomic_number=atomic_number,
        pbc=pbc
    )

    return state


@pytest.fixture
def mock_context():
    """Create a simple mock SimulationContext object.
    创建一个简单的模拟 SimulationContext 对象。

    Returns:
        SimulationContext: A context with target temperature 300K and no constraints.
            具有目标温度 300K 且无约束的环境。
    """
    context = SimulationContext(
        target_temp=300.0,
        target_pressure=None,
        constraints=[]
    )
    return context


@pytest.fixture
def mock_state_with_momentum(mock_state):
    """Create a state with non-zero momentum for kinetic energy tests.
    为动能测试创建具有非零动量的状态。

    Returns:
        State: A state with known momenta, useful for KE calculations.
            具有已知动量的状态，对 KE 计算很有用。
    """
    array_module = cp if HAS_CUPY else np

    # Set specific momenta: p_i = [0.01, 0.01, 0.01] in amu*Ang/ps
    # 设置特定的动量：p_i = [0.01, 0.01, 0.01] 在 amu*Ang/ps 中
    mock_state.p = array_module.array([
        [0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01]
    ], dtype=float)

    return mock_state
