"""Velocity and configuration initializer tests.
速度和配置初始化程序测试。

Tests verify that initializers produce correct distributions and statistics,
particularly the Maxwell-Boltzmann distribution.
"""

import pytest
import numpy as np
import math

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from pydyn.initializer import MaxwellBoltzmannDistribution
from pydyn.state import State
from pydyn.context import SimulationContext
from pydyn.constants import Constants


class TestMaxwellBoltzmannInitializer:
    """Test the Maxwell-Boltzmann velocity initializer.
    测试 Maxwell-Boltzmann 速度初始化程序。
    """

    def test_initializer_creation(self):
        """Test that initializer can be created.
        测试可以创建初始化程序。
        """
        initializer = MaxwellBoltzmannDistribution(target_temp=300.0)
        assert initializer is not None
        assert initializer.target_temp == 300.0

    def test_initializer_default_temp(self):
        """Test that initializer can use context temperature.
        测试初始化程序可以使用上下文温度。
        """
        initializer = MaxwellBoltzmannDistribution()
        assert initializer.target_temp is None

    def test_initialize_sets_momenta(self, mock_state):
        """Test that initialize sets momenta (not zero).
        测试初始化设置动量（不是零）。
        """
        array_module = cp if HAS_CUPY else np

        initializer = MaxwellBoltzmannDistribution(target_temp=300.0)
        context = SimulationContext(target_temp=300.0)

        # Momenta should be zero before
        # 之前动量应该为零
        assert array_module.allclose(mock_state.p, array_module.zeros((3, 3)))

        # Initialize
        # 初始化
        initializer.initialize(mock_state, context)

        # Momenta should now be non-zero
        # 动量现在应该非零
        assert not array_module.allclose(mock_state.p, array_module.zeros((3, 3)))

    def test_distribution_statistics(self):
        """Test that initialized momenta have correct statistical properties.
        测试初始化的动量具有正确的统计性质。

        For a Maxwell-Boltzmann distribution at temperature T:
        - Mean should be near zero
        - Variance should match sigma_p = sqrt(m * k_B * T * e_to_mv2)
        / 对于温度 T 下的 Maxwell-Boltzmann 分布：
        - 均值应接近零
        - 方差应匹配 sigma_p = sqrt(m * k_B * T * e_to_mv2)
        """
        array_module = cp if HAS_CUPY else np

        # Create a large state for better statistics
        # 创建一个大的状态以获得更好的统计
        N = 1000
        r = array_module.zeros((N, 3), dtype=float)
        p = array_module.zeros((N, 3), dtype=float)
        m = array_module.ones(N, dtype=float)  # All mass = 1 amu
        # 所有质量 = 1 amu
        box = array_module.eye(3, dtype=float) * 100
        atomic_number = array_module.array([18] * N, dtype=int)
        pbc = array_module.array([True, True, True])

        state = State(r=r, p=p, m=m, box=box, atomic_number=atomic_number, pbc=pbc)

        target_temp = 300.0
        initializer = MaxwellBoltzmannDistribution(target_temp=target_temp)
        context = SimulationContext(target_temp=target_temp)

        initializer.initialize(state, context)

        # Expected variance for each momentum component
        # 每个动量分量的预期方差
        sigma_p = math.sqrt(
            Constants.kB * target_temp * Constants.e_to_mv2
        )
        variance_expected = sigma_p ** 2

        # Get momenta as numpy for statistics
        # 获取动量作为 numpy 进行统计
        p_np = np.asarray(state.p)

        # Mean should be near zero (within 5% for large sample)
        # 均值应接近零（大样本为 5% 以内）
        mean = np.mean(p_np)
        assert abs(mean) < 0.05 * sigma_p, \
            f"Mean momentum {mean} far from zero (sigma_p={sigma_p})"

        # Variance should match expected (within 10%)
        # 方差应匹配预期（10% 以内）
        variance_actual = np.var(p_np)
        relative_error = abs(variance_actual - variance_expected) / variance_expected
        assert relative_error < 0.1, \
            f"Variance mismatch: expected {variance_expected}, got {variance_actual}"

    def test_temperature_affects_distribution(self):
        """Test that higher temperature gives larger momenta.
        测试更高的温度给出更大的动量。
        """
        array_module = cp if HAS_CUPY else np

        N = 100
        r = array_module.zeros((N, 3), dtype=float)
        p = array_module.zeros((N, 3), dtype=float)
        m = array_module.ones(N, dtype=float)
        box = array_module.eye(3, dtype=float) * 100
        atomic_number = array_module.array([18] * N, dtype=int)
        pbc = array_module.array([True, True, True])

        # Initialize at low temperature
        # 在低温下初始化
        state_low = State(r=r.copy(), p=p.copy(), m=m.copy(), box=box.copy(),
                          atomic_number=atomic_number, pbc=pbc)
        initializer_low = MaxwellBoltzmannDistribution(target_temp=100.0)
        context = SimulationContext()
        initializer_low.initialize(state_low, context)

        # Initialize at high temperature
        # 在高温下初始化
        state_high = State(r=r.copy(), p=p.copy(), m=m.copy(), box=box.copy(),
                           atomic_number=atomic_number, pbc=pbc)
        initializer_high = MaxwellBoltzmannDistribution(target_temp=500.0)
        initializer_high.initialize(state_high, context)

        # Higher temperature should give larger momentum magnitudes (on average)
        # 更高的温度应该给出更大的动量大小（平均）
        p_low_np = np.asarray(state_low.p)
        p_high_np = np.asarray(state_high.p)

        rms_low = np.sqrt(np.mean(p_low_np ** 2))
        rms_high = np.sqrt(np.mean(p_high_np ** 2))

        assert rms_high > rms_low, \
            f"Higher temp should give larger momenta: {rms_high} vs {rms_low}"

    def test_mass_affects_variance(self):
        """Test that variance scales with mass as expected.
        测试方差按预期与质量缩放。

        sigma_p ∝ sqrt(m * T), so heavier atoms have larger momentum variance.
        / sigma_p ∝ sqrt(m * T)，所以较重的原子有更大的动量方差。
        """
        array_module = cp if HAS_CUPY else np

        N = 100
        r = array_module.zeros((N, 3), dtype=float)
        p = array_module.zeros((N, 3), dtype=float)
        box = array_module.eye(3, dtype=float) * 100
        atomic_number = array_module.array([18] * N, dtype=int)
        pbc = array_module.array([True, True, True])

        target_temp = 300.0

        # Light atoms
        # 轻原子
        state_light = State(
            r=r.copy(), p=p.copy(), m=array_module.ones(N, dtype=float),
            box=box.copy(), atomic_number=atomic_number, pbc=pbc
        )
        initializer = MaxwellBoltzmannDistribution(target_temp=target_temp)
        context = SimulationContext()
        initializer.initialize(state_light, context)

        # Heavy atoms
        # 重原子
        state_heavy = State(
            r=r.copy(), p=p.copy(), m=array_module.ones(N, dtype=float) * 10,
            box=box.copy(), atomic_number=atomic_number, pbc=pbc
        )
        initializer.initialize(state_heavy, context)

        # Heavy atoms should have ~sqrt(10) times larger momentum variance
        # 重原子应该有 ~sqrt(10) 倍的动量方差
        p_light_np = np.asarray(state_light.p)
        p_heavy_np = np.asarray(state_heavy.p)

        var_light = np.var(p_light_np)
        var_heavy = np.var(p_heavy_np)

        expected_ratio = math.sqrt(10.0)
        actual_ratio = math.sqrt(var_heavy / var_light)

        # Allow 20% error due to sampling
        # 由于采样允许 20% 的误差
        assert abs(actual_ratio - expected_ratio) < 0.2 * expected_ratio, \
            f"Mass scaling wrong: expected {expected_ratio}, got {actual_ratio}"

    def test_context_temperature_fallback(self, mock_state):
        """Test that initializer uses context temp when own temp is None.
        测试初始化程序在自己的温度为 None 时使用上下文温度。
        """
        initializer = MaxwellBoltzmannDistribution(target_temp=None)
        context = SimulationContext(target_temp=400.0)

        # Should not raise
        # 不应该抛出异常
        initializer.initialize(mock_state, context)

        # Momenta should be set
        # 动量应该被设置
        array_module = cp if HAS_CUPY else np
        assert not array_module.allclose(mock_state.p, array_module.zeros((3, 3)))

    def test_kinetic_energy_matches_target_temperature(self):
        """Test that initialized KE roughly matches target temperature.
        测试初始化的 KE 大致匹配目标温度。

        Using equipartition: KE ≈ 0.5 * (3*N) * k_B * T
        / 使用等分配：KE ≈ 0.5 * (3*N) * k_B * T
        """
        array_module = cp if HAS_CUPY else np

        N = 100
        r = array_module.zeros((N, 3), dtype=float)
        p = array_module.zeros((N, 3), dtype=float)
        m = array_module.ones(N, dtype=float)
        box = array_module.eye(3, dtype=float) * 100
        atomic_number = array_module.array([18] * N, dtype=int)
        pbc = array_module.array([True, True, True])

        state = State(r=r, p=p, m=m, box=box, atomic_number=atomic_number, pbc=pbc)

        target_temp = 300.0
        initializer = MaxwellBoltzmannDistribution(target_temp=target_temp)
        context = SimulationContext()
        initializer.initialize(state, context)

        # Expected KE from equipartition (3 DOF per atom)
        # 从等分配的预期 KE（每个原子 3 个 DOF）
        expected_ke = 1.5 * N * Constants.kB * target_temp
        actual_ke = state.kinetic_energy

        # Allow 10% error due to sampling
        # 由于采样允许 10% 的误差
        relative_error = abs(actual_ke - expected_ke) / expected_ke
        assert relative_error < 0.1, \
            f"KE mismatch: expected {expected_ke}, got {actual_ke}"
