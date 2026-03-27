"""Ensemble integration tests.
系综积分测试。

Tests verify ensemble structure, operator composition, and integration steps
for various thermodynamic ensembles.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from pydyn.ensembles.verlet import VelocityVerlet
from pydyn.ensembles.base import PositionOp, MomentumOp, Ensemble
from pydyn.state import State
from pydyn.context import SimulationContext
from pydyn.forces.base import ForceModel


class MockForceModel(ForceModel):
    """Mock force model for testing.
    用于测试的模拟力模型。
    """
    implemented_properties = ["forces"]

    def compute(self, state, context, properties=None):
        """Return small dummy forces.
        返回小的虚拟力。
        """
        array_module = cp if HAS_CUPY else np
        # Small constant force for testing
        # 小的恒定力用于测试
        self.results["forces"] = array_module.ones_like(state.r) * 0.001
        self.state = state.copy()


class TestVelocityVerletStructure:
    """Test Velocity Verlet ensemble structure.
    测试 Velocity Verlet 系综结构。
    """

    def test_op_list_structure(self):
        """Test that op_list has correct structure.
        测试 op_list 具有正确的结构。

        Should be: [(MomentumOp, 0.5), (PositionOp, 1.0), (MomentumOp, 0.5)]
        / 应该是：[(MomentumOp, 0.5), (PositionOp, 1.0), (MomentumOp, 0.5)]
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        assert hasattr(ensemble, 'op_list')
        assert len(ensemble.op_list) == 3

        # Check structure: (operator, timestep_fraction) tuples
        # 检查结构：(operator, timestep_fraction) 元组
        for item in ensemble.op_list:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_op_list_operators(self):
        """Test that op_list contains correct operator types.
        测试 op_list 包含正确的操作符类型。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        op0, ts0 = ensemble.op_list[0]
        op1, ts1 = ensemble.op_list[1]
        op2, ts2 = ensemble.op_list[2]

        assert isinstance(op0, MomentumOp)
        assert isinstance(op1, PositionOp)
        assert isinstance(op2, MomentumOp)

    def test_op_list_timesteps(self):
        """Test that op_list has correct timestep fractions.
        测试 op_list 具有正确的时间步分数。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        timesteps = [ts for _, ts in ensemble.op_list]
        expected = [0.5, 1.0, 0.5]

        assert timesteps == expected

    def test_force_model_stored(self):
        """Test that force model is accessible in operators.
        测试力模型在操作符中是可访问的。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        momentum_ops = [op for op, _ in ensemble.op_list if isinstance(op, MomentumOp)]
        assert len(momentum_ops) == 2

        for op in momentum_ops:
            assert op.force_model is force_model


class TestEnsembleStep:
    """Test ensemble stepping.
    测试系综步进。
    """

    def test_step_runs_without_error(self, mock_state, mock_context):
        """Test that step() runs without raising exceptions.
        测试 step() 运行不会抛出异常。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        # Should not raise
        # 不应该抛出
        ensemble.step(mock_state, mock_context, dt=0.001)

    def test_step_with_no_constraints(self, mock_state, mock_context):
        """Test step with empty constraint list.
        测试没有约束列表的步。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        array_module = cp if HAS_CUPY else np
        r_before = mock_state.r.copy()

        ensemble.step(mock_state, mock_context, dt=0.001)

        # Position should have changed (at least first position update)
        # 位置应该改变（至少第一次位置更新）
        assert not array_module.allclose(mock_state.r, r_before, atol=1e-10)

    def test_step_modifies_state(self, mock_state_with_momentum, mock_context):
        """Test that step modifies state.
        测试步修改状态。
        """
        state = mock_state_with_momentum
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        array_module = cp if HAS_CUPY else np
        p_before = state.p.copy()

        ensemble.step(state, mock_context, dt=0.001)

        # Momentum should have changed due to forces
        # 动量应该因为力而改变
        assert not array_module.allclose(state.p, p_before, atol=1e-10)

    def test_step_respects_constraints(self, mock_state, mock_context):
        """Test that constraints are applied after step.
        测试约束在步之后应用。
        """
        from pydyn.constraints import RemoveCOMMomentum

        state = mock_state
        context = SimulationContext(
            target_temp=300.0,
            constraints=[RemoveCOMMomentum()]
        )

        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        # Apply initial momentum
        # 应用初始动量
        array_module = cp if HAS_CUPY else np
        state.p[:] = array_module.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=float)

        ensemble.step(state, context, dt=0.001)

        # After step with RemoveCOMMomentum constraint, total momentum should be zero
        # 应用 RemoveCOMMomentum 约束的步后，总动量应该为零
        total_p = array_module.sum(state.p, axis=0)
        assert array_module.allclose(total_p, array_module.zeros(3), atol=1e-10)


class TestPositionOperator:
    """Test PositionOp operator.
    测试 PositionOp 操作符。
    """

    def test_position_update(self, mock_state_with_momentum):
        """Test that position is updated correctly.
        测试位置正确更新。

        r' = r + dt * (p / m)
        / r' = r + dt * (p / m)
        """
        state = mock_state_with_momentum
        array_module = cp if HAS_CUPY else np

        r_before = state.r.copy()
        p = state.p
        m = state.m

        op = PositionOp()
        op.apply(state, None, dt=0.001)

        # Check first position update: r' = r + dt * p / m
        # 检查第一个位置更新：r' = r + dt * p / m
        expected_r = r_before + 0.001 * p / m[:, None]
        assert array_module.allclose(state.r, expected_r, atol=1e-10)

    def test_position_zero_momentum(self, mock_state):
        """Test that zero momentum gives no position change.
        测试零动量不改变位置。
        """
        state = mock_state
        array_module = cp if HAS_CUPY else np

        r_before = state.r.copy()

        op = PositionOp()
        op.apply(state, None, dt=0.001)

        # Position should be unchanged
        # 位置应该不变
        assert array_module.allclose(state.r, r_before)


class TestMomentumOperator:
    """Test MomentumOp operator.
    测试 MomentumOp 操作符。
    """

    def test_momentum_update(self, mock_state):
        """Test that momentum is updated from forces.
        测试动量从力更新。

        p' = p + dt * F * e_to_mv2
        / p' = p + dt * F * e_to_mv2
        """
        from pydyn.constants import Constants

        state = mock_state
        array_module = cp if HAS_CUPY else np

        # Set specific initial momentum
        # 设置特定的初始动量
        state.p[:] = array_module.array([
            [0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=float)

        p_before = state.p.copy()

        force_model = MockForceModel()
        op = MomentumOp(force_model)

        dt = 0.001
        op.apply(state, None, dt=dt)

        # Forces from MockForceModel are 0.001 N everywhere
        # 来自 MockForceModel 的力在各处都是 0.001 N
        # p' = p + dt * F * e_to_mv2
        expected_dp = dt * 0.001 * Constants.e_to_mv2
        expected_p = p_before + expected_dp

        assert array_module.allclose(state.p, expected_p, atol=1e-10)

    def test_momentum_zero_force(self, mock_state):
        """Test that zero force gives no momentum change.
        测试零力不改变动量。
        """
        class ZeroForceModel(ForceModel):
            def compute(self, state, context, properties=None):
                array_module = cp if HAS_CUPY else np
                self.results["forces"] = array_module.zeros_like(state.r)
                self.state = state.copy()

        state = mock_state
        array_module = cp if HAS_CUPY else np

        state.p[:] = array_module.array([
            [0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=float)

        p_before = state.p.copy()

        force_model = ZeroForceModel()
        op = MomentumOp(force_model)
        op.apply(state, None, dt=0.001)

        # Momentum should be unchanged
        # 动量应该不变
        assert array_module.allclose(state.p, p_before)


class TestEnsembleConservedEnergy:
    """Test ensemble conserved energy methods.
    测试系综守恒能量方法。
    """

    def test_get_conserved_energy_default_none(self):
        """Test that default ensemble returns None for conserved energy.
        测试默认系综对守恒能量返回 None。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        energy = ensemble.get_conserved_energy(None, None)
        assert energy is None

    def test_get_pressure_default_none(self):
        """Test that default ensemble returns None for pressure.
        测试默认系综对压力返回 None。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        pressure = ensemble.get_pressure(None, None)
        assert pressure is None
