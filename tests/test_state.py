"""State class and extension tests.
State 类和扩展测试。

Tests verify state management, kinetic energy calculations, volume computation,
extension registry, and roundtrip conversions with ASE Atoms objects.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from pydyn.state import State, SpinExtension, NHThermostatExtension, MTTKBarostatExtension
from pydyn.constants import Constants


class TestStateInitialization:
    """Test State object creation and basic properties.
    测试 State 对象创建和基本属性。
    """

    def test_state_creation(self, mock_state):
        """Test that a state can be created with all attributes.
        测试可以使用所有属性创建状态。
        """
        assert mock_state is not None
        assert mock_state.N == 3
        assert mock_state.r is not None
        assert mock_state.p is not None
        assert mock_state.m is not None
        assert mock_state.box is not None

    def test_state_atom_count(self, mock_state):
        """Test that N reflects the actual number of atoms.
        测试 N 反映实际原子数。
        """
        assert mock_state.N == len(mock_state.r)
        assert mock_state.N == len(mock_state.m)
        assert mock_state.N == len(mock_state.p)

    def test_empty_state_initialization(self):
        """Test that a state with no atoms can be created.
        测试可以创建没有原子的状态。
        """
        state = State()
        assert state.N == 0
        assert state.r is None


class TestKineticEnergy:
    """Test kinetic energy calculations.
    测试动能计算。
    """

    def test_zero_momentum_zero_ke(self, mock_state):
        """Test that zero momentum gives zero kinetic energy.
        测试零动量给出零动能。
        """
        assert mock_state.kinetic_energy == 0.0

    def test_kinetic_energy_calculation(self, mock_state_with_momentum):
        """Test kinetic energy is correctly computed from momenta.
        测试从动量正确计算动能。

        KE = 0.5 * sum(p_i^2 / m_i) * mv2_to_e
        / KE = 0.5 * sum(p_i^2 / m_i) * mv2_to_e
        """
        state = mock_state_with_momentum
        # p = [0.01, 0.01, 0.01] for each atom, m = 1.0
        # 每个原子 p = [0.01, 0.01, 0.01]，m = 1.0
        # KE = 0.5 * 3 atoms * (0.01^2 + 0.01^2 + 0.01^2) / 1.0 * mv2_to_e
        # KE = 0.5 * 3 * 0.0003 * mv2_to_e

        array_module = cp if HAS_CUPY else np
        expected_ke_raw = 0.5 * 3 * (0.01**2 + 0.01**2 + 0.01**2) / 1.0
        expected_ke = expected_ke_raw * Constants.mv2_to_e

        # Allow for floating point precision
        # 允许浮点精度
        assert abs(state.kinetic_energy - expected_ke) < 1e-12

    def test_kinetic_energy_positive(self, mock_state_with_momentum):
        """Test that kinetic energy is non-negative.
        测试动能为非负。
        """
        assert mock_state_with_momentum.kinetic_energy >= 0.0

    def test_kinetic_energy_proportional_to_mass(self):
        """Test that heavier atoms contribute more to KE.
        测试较重的原子对 KE 贡献更多。
        """
        array_module = cp if HAS_CUPY else np

        # Create two states with same momentum but different masses
        # 创建两个状态，相同动量但不同质量
        p_common = 0.01

        # Light atom (m=1)
        state1 = State(
            r=array_module.array([[0.0, 0.0, 0.0]], dtype=float),
            p=array_module.array([[p_common, 0.0, 0.0]], dtype=float),
            m=array_module.array([1.0], dtype=float),
            box=array_module.eye(3, dtype=float) * 10,
            atomic_number=array_module.array([18], dtype=int),
            pbc=array_module.array([True, True, True])
        )

        # Heavy atom (m=10)
        state2 = State(
            r=array_module.array([[0.0, 0.0, 0.0]], dtype=float),
            p=array_module.array([[p_common, 0.0, 0.0]], dtype=float),
            m=array_module.array([10.0], dtype=float),
            box=array_module.eye(3, dtype=float) * 10,
            atomic_number=array_module.array([18], dtype=int),
            pbc=array_module.array([True, True, True])
        )

        # Same momentum, different mass -> KE inversely proportional to mass
        # 相同动量，不同质量 -> KE 与质量成反比
        assert state1.kinetic_energy > state2.kinetic_energy


class TestVolume:
    """Test volume calculations.
    测试体积计算。
    """

    def test_volume_cubic_cell(self, mock_state):
        """Test volume of a cubic cell.
        测试立方体盒的体积。
        """
        # Default box is 10x10x10 Angstroms
        # 默认盒是 10x10x10 Angstroms
        expected_volume = 10.0 * 10.0 * 10.0
        assert abs(mock_state.volume - expected_volume) < 1e-10

    def test_volume_zero_for_none_box(self):
        """Test that volume is zero if box is None.
        测试如果盒为 None，体积为零。
        """
        state = State()
        assert state.volume == 0.0

    def test_volume_identity_determinant(self):
        """Test volume for identity box matrix.
        测试单位矩阵盒的体积。
        """
        array_module = cp if HAS_CUPY else np
        state = State(
            r=array_module.array([[0.0, 0.0, 0.0]], dtype=float),
            p=array_module.zeros((1, 3), dtype=float),
            m=array_module.array([1.0], dtype=float),
            box=array_module.eye(3, dtype=float),
            atomic_number=array_module.array([18], dtype=int),
            pbc=array_module.array([True, True, True])
        )
        assert abs(state.volume - 1.0) < 1e-10


class TestExtensionRegistry:
    """Test state extension registration and retrieval.
    测试状态扩展注册和检索。
    """

    def test_register_extension(self, mock_state):
        """Test that extensions can be registered.
        测试可以注册扩展。
        """
        extension = SpinExtension()
        mock_state.register_extension("test_ext", extension)
        assert mock_state.has_extension("test_ext")

    def test_get_extension(self, mock_state):
        """Test that registered extensions can be retrieved.
        测试可以检索已注册的扩展。
        """
        extension = SpinExtension()
        mock_state.register_extension("spin", extension)
        retrieved = mock_state.get_extension("spin")
        assert retrieved is extension

    def test_get_missing_extension_raises(self, mock_state):
        """Test that retrieving missing extension raises KeyError.
        测试检索缺失的扩展会引发 KeyError。
        """
        with pytest.raises(KeyError):
            mock_state.get_extension("nonexistent")

    def test_extension_empty_name_raises(self, mock_state):
        """Test that registering with empty name raises ValueError.
        测试使用空名称注册会引发 ValueError。
        """
        extension = SpinExtension()
        with pytest.raises(ValueError):
            mock_state.register_extension("", extension)

    def test_extension_none_raises(self, mock_state):
        """Test that registering None extension raises ValueError.
        测试注册 None 扩展会引发 ValueError。
        """
        with pytest.raises(ValueError):
            mock_state.register_extension("bad", None)

    def test_has_extension_false_for_unregistered(self, mock_state):
        """Test that has_extension returns False for unregistered.
        测试 has_extension 对于未注册的返回 False。
        """
        assert not mock_state.has_extension("nonexistent")

    def test_multiple_extensions(self, mock_state):
        """Test registering multiple extensions.
        测试注册多个扩展。
        """
        spin_ext = SpinExtension()
        nh_ext = NHThermostatExtension()
        mttk_ext = MTTKBarostatExtension()

        mock_state.register_extension("spin", spin_ext)
        mock_state.register_extension("nh", nh_ext)
        mock_state.register_extension("mttk", mttk_ext)

        assert mock_state.has_extension("spin")
        assert mock_state.has_extension("nh")
        assert mock_state.has_extension("mttk")


class TestStateCopy:
    """Test state copying and independence.
    测试状态复制和独立性。
    """

    def test_copy_creates_independent_state(self, mock_state_with_momentum):
        """Test that copy creates independent arrays.
        测试复制创建独立的数组。
        """
        state_copy = mock_state_with_momentum.copy()

        # Modify the copy's momentum
        # 修改副本的动量
        array_module = cp if HAS_CUPY else np
        state_copy.p[0, 0] = 999.0

        # Original should be unchanged
        # 原始应该保持不变
        original_val = float(mock_state_with_momentum.p[0, 0])
        assert original_val != 999.0

    def test_copy_has_same_values(self, mock_state_with_momentum):
        """Test that copy has same values as original.
        测试副本与原始值相同。
        """
        state_copy = mock_state_with_momentum.copy()

        array_module = cp if HAS_CUPY else np
        assert array_module.allclose(state_copy.r, mock_state_with_momentum.r)
        assert array_module.allclose(state_copy.p, mock_state_with_momentum.p)
        assert array_module.allclose(state_copy.m, mock_state_with_momentum.m)

    def test_copy_preserves_extensions(self, mock_state):
        """Test that copying preserves registered extensions.
        测试复制保留已注册的扩展。
        """
        extension = SpinExtension()
        mock_state.register_extension("spin", extension)

        state_copy = mock_state.copy()
        assert state_copy.has_extension("spin")


class TestConfigureSameAs:
    """Test configuration comparison method.
    测试配置比较方法。
    """

    def test_identical_states_match(self, mock_state):
        """Test that identical states match.
        测试相同的状态匹配。
        """
        assert mock_state.configure_same_as(mock_state)

    def test_different_atom_counts_dont_match(self, mock_state):
        """Test that states with different N don't match.
        测试不同 N 的状态不匹配。
        """
        array_module = cp if HAS_CUPY else np
        state2 = State(
            r=array_module.array([[0.0, 0.0, 0.0]], dtype=float),
            p=array_module.zeros((1, 3), dtype=float),
            m=array_module.array([1.0], dtype=float),
            box=array_module.eye(3, dtype=float) * 10,
            atomic_number=array_module.array([18], dtype=int),
            pbc=array_module.array([True, True, True])
        )
        assert not mock_state.configure_same_as(state2)

    def test_different_positions_dont_match(self, mock_state):
        """Test that states with different positions don't match.
        测试具有不同位置的状态不匹配。
        """
        array_module = cp if HAS_CUPY else np
        state_copy = mock_state.copy()
        state_copy.r[0, 0] += 1.0
        assert not mock_state.configure_same_as(state_copy)


class TestKineticVirial:
    """Test kinetic virial tensor calculations.
    测试动能维里张量计算。
    """

    def test_kinetic_virial_shape(self, mock_state_with_momentum):
        """Test that kinetic virial is 3x3.
        测试动能维里是 3x3。
        """
        virial = mock_state_with_momentum.kinetic_virial
        assert virial.shape == (3, 3)

    def test_kinetic_virial_zero_for_zero_momentum(self, mock_state):
        """Test that kinetic virial is zero for zero momentum.
        测试零动量时动能维里为零。
        """
        array_module = cp if HAS_CUPY else np
        virial = mock_state.kinetic_virial
        assert array_module.allclose(virial, array_module.zeros((3, 3)))
