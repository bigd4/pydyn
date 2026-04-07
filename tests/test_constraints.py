"""Constraint tests.
约束测试。

Tests verify constraint behavior including center-of-mass momentum removal
and degree-of-freedom accounting.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from pydyn.constraints import RemoveCOMMomentum
from pydyn.state import State
from pydyn.context import SimulationContext


class TestRemoveCOMMomentum:
    """Test the RemoveCOMMomentum constraint.
    测试 RemoveCOMMomentum 约束。
    """

    def test_removed_dof_is_three(self):
        """Test that removed_dof equals 3.
        测试 removed_dof 等于 3。
        """
        constraint = RemoveCOMMomentum()
        assert constraint.removed_dof == 3

    def test_apply_zeros_total_momentum(self, mock_state_with_momentum):
        """Test that applying constraint zeros the total momentum.
        测试应用约束将总动量归零。
        """
        array_module = cp if HAS_CUPY else np

        # Create a state with non-zero total momentum
        # 创建总动量非零的状态
        state = mock_state_with_momentum

        # Before constraint: total momentum should be non-zero
        # 约束前：总动量应该是非零的
        total_momentum_before = array_module.sum(state.p, axis=0)
        assert array_module.linalg.norm(total_momentum_before) > 1e-10

        # Apply constraint
        # 应用约束
        constraint = RemoveCOMMomentum()
        context = SimulationContext()
        constraint.apply(state, context)

        # After constraint: total momentum should be zero
        # 约束后：总动量应该为零
        total_momentum_after = array_module.sum(state.p, axis=0)
        assert array_module.allclose(total_momentum_after, array_module.zeros(3), atol=1e-10)

    def test_apply_with_different_masses(self):
        """Test constraint works with non-uniform masses.
        测试约束适用于非均匀质量。
        """
        array_module = cp if HAS_CUPY else np

        # Create a state with different masses
        # 创建不同质量的状态
        state = State(
            r=array_module.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0]
            ], dtype=float),
            p=array_module.array([
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0]
            ], dtype=float),
            m=array_module.array([1.0, 2.0, 3.0], dtype=float),
            box=array_module.eye(3, dtype=float) * 10,
            atomic_number=array_module.array([18, 18, 18], dtype=int),
            pbc=array_module.array([True, True, True])
        )

        # Apply constraint
        # 应用约束
        constraint = RemoveCOMMomentum()
        context = SimulationContext()
        constraint.apply(state, context)

        # Total momentum should be zero
        # 总动量应该为零
        total_momentum = array_module.sum(state.p, axis=0)
        assert array_module.allclose(total_momentum, array_module.zeros(3), atol=1e-10)

    def test_constraint_preserves_kinetic_energy_shape(self, mock_state_with_momentum):
        """Test that constraint maintains KE calculation capability.
        测试约束保持 KE 计算能力。
        """
        state = mock_state_with_momentum
        constraint = RemoveCOMMomentum()
        context = SimulationContext()

        ke_before = state.kinetic_energy
        constraint.apply(state, context)
        ke_after = state.kinetic_energy

        # Both should be computable (non-NaN)
        # 两者都应该是可计算的（非 NaN）
        assert not np.isnan(ke_before)
        assert not np.isnan(ke_after)

    def test_zero_momentum_unchanged(self, mock_state):
        """Test that zero momentum state is unchanged by constraint.
        测试零动量状态不被约束改变。
        """
        array_module = cp if HAS_CUPY else np

        state = mock_state
        state_copy = state.copy()

        constraint = RemoveCOMMomentum()
        context = SimulationContext()
        constraint.apply(state, context)

        # Should be unchanged (zero momentum stays zero)
        # 应该保持不变（零动量保持零）
        assert array_module.allclose(state.p, state_copy.p)

    def test_multiple_applications_idempotent(self, mock_state_with_momentum):
        """Test that applying constraint multiple times is idempotent.
        测试多次应用约束是幂等的。
        """
        array_module = cp if HAS_CUPY else np

        state = mock_state_with_momentum
        constraint = RemoveCOMMomentum()
        context = SimulationContext()

        # Apply once
        # 应用一次
        constraint.apply(state, context)
        momentum_after_first = state.p.copy()

        # Apply again
        # 再次应用
        constraint.apply(state, context)
        momentum_after_second = state.p.copy()

        # Should be the same (since total momentum was already zero)
        # 应该相同（因为总动量已经为零）
        assert array_module.allclose(momentum_after_first, momentum_after_second)

    def test_com_velocity_calculation(self):
        """Test that COM velocity is correctly calculated.
        测试质心速度计算正确。
        """
        array_module = cp if HAS_CUPY else np

        # Create a simple state with known velocities
        # 创建具有已知速度的简单状态
        state = State(
            r=array_module.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ], dtype=float),
            p=array_module.array([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ], dtype=float),
            m=array_module.array([1.0, 1.0], dtype=float),
            box=array_module.eye(3, dtype=float) * 10,
            atomic_number=array_module.array([18, 18], dtype=int),
            pbc=array_module.array([True, True, True])
        )

        # Total momentum should be [2, 0, 0]
        # 总动量应该是 [2, 0, 0]
        total_p = array_module.sum(state.p, axis=0)
        assert array_module.allclose(total_p, array_module.array([2.0, 0.0, 0.0]))

        # Total mass is 2
        # 总质量是 2
        total_m = array_module.sum(state.m)
        assert total_m == 2.0

        # COM velocity should be p_total / m_total = [1, 0, 0]
        # 质心速度应该是 p_total / m_total = [1, 0, 0]
        v_com_expected = total_p / total_m

        constraint = RemoveCOMMomentum()
        context = SimulationContext()
        constraint.apply(state, context)

        # After constraint, total momentum should be zero
        # 约束后，总动量应该为零
        assert array_module.allclose(array_module.sum(state.p, axis=0), array_module.zeros(3), atol=1e-10)
