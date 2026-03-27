"""Simulation runner tests.
模拟运行器测试。

Tests verify simulation initialization, stepping, running, and finalization,
including interaction with initializers, ensembles, and observers.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from pydyn.simulation import Simulation
from pydyn.state import State
from pydyn.context import SimulationContext
from pydyn.initializer import MaxwellBoltzmannDistribution
from pydyn.ensembles.verlet import VelocityVerlet
from pydyn.forces.base import ForceModel


class MockForceModel(ForceModel):
    """Mock force model for testing.
    用于测试的模拟力模型。
    """
    implemented_properties = ["forces"]

    def compute(self, state, context, properties=None):
        """Return dummy forces (zeros).
        返回虚拟力（零）。
        """
        array_module = cp if HAS_CUPY else np
        self.results["forces"] = array_module.zeros_like(state.r)
        self.state = state.copy()


class MockObserver:
    """Mock observer for tracking simulation callbacks.
    用于跟踪模拟回调的模拟观察者。
    """
    def __init__(self):
        """Initialize observer.
        初始化观察者。
        """
        self.call_count = 0
        self.initialized = False
        self.finalized = False

    def initialize(self):
        """Track initialization.
        跟踪初始化。
        """
        self.initialized = True

    def __call__(self, sim):
        """Track calls.
        跟踪调用。
        """
        self.call_count += 1

    def finalize(self):
        """Track finalization.
        跟踪最终化。
        """
        self.finalized = True


class TestSimulationInitialization:
    """Test Simulation object creation and setup.
    测试 Simulation 对象创建和设置。
    """

    def test_simulation_creation(self, mock_state, mock_context):
        """Test that a simulation can be created.
        测试可以创建模拟。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        initializer = [MaxwellBoltzmannDistribution(target_temp=300.0)]

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=initializer,
            ensemble=ensemble
        )

        assert sim is not None
        assert sim.state is mock_state
        assert sim.context is mock_context
        assert sim.dt == 0.001

    def test_simulation_attributes(self, mock_state, mock_context):
        """Test that simulation has correct initial attributes.
        测试模拟具有正确的初始属性。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.002,
            initializer=[],
            ensemble=ensemble
        )

        assert sim.time == 0.0
        assert sim.step_count == 0
        assert len(sim.observers) == 0

    def test_simulation_with_observers(self, mock_state, mock_context):
        """Test that simulation can be created with observers.
        测试可以使用观察者创建模拟。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        observer1 = MockObserver()
        observer2 = MockObserver()

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble,
            observers=[observer1, observer2]
        )

        assert len(sim.observers) == 2


class TestSimulationInitialize:
    """Test simulation initialization phase.
    测试模拟初始化阶段。
    """

    def test_initialize_calls_initializers(self, mock_state, mock_context):
        """Test that initialize() calls all initializers.
        测试 initialize() 调用所有初始化程序。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        init = MaxwellBoltzmannDistribution(target_temp=300.0)

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[init],
            ensemble=ensemble
        )

        array_module = cp if HAS_CUPY else np

        # Before initialize, momentum should be zero
        # 初始化前，动量应该为零
        assert array_module.allclose(mock_state.p, array_module.zeros((3, 3)))

        sim.initialize()

        # After initialize, momentum should be set by initializer
        # 初始化后，初始化程序应该设置动量
        assert not array_module.allclose(mock_state.p, array_module.zeros((3, 3)))

    def test_initialize_calls_observers(self, mock_state, mock_context):
        """Test that initialize() calls observer.initialize().
        测试 initialize() 调用 observer.initialize()。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        observer = MockObserver()

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble,
            observers=[observer]
        )

        assert not observer.initialized
        sim.initialize()
        assert observer.initialized

    def test_initialize_idempotent(self, mock_state, mock_context):
        """Test that calling initialize multiple times is safe.
        测试多次调用初始化是安全的。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        observer = MockObserver()

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble,
            observers=[observer]
        )

        sim.initialize()
        sim.initialize()
        # Should have called initialize() only once
        # 应该只调用一次 initialize()
        assert observer.initialized


class TestSimulationStep:
    """Test simulation stepping.
    测试模拟步进。
    """

    def test_step_increments_counters(self, mock_state, mock_context):
        """Test that step() increments time and step_count.
        测试 step() 增加时间和步数计数。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        dt = 0.001

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=dt,
            initializer=[],
            ensemble=ensemble
        )

        assert sim.time == 0.0
        assert sim.step_count == 0

        sim.step()

        assert abs(sim.time - dt) < 1e-15
        assert sim.step_count == 1

    def test_step_auto_initializes(self, mock_state, mock_context):
        """Test that step() calls initialize if not already done.
        测试 step() 在尚未初始化时调用初始化。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        init = MaxwellBoltzmannDistribution(target_temp=300.0)

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[init],
            ensemble=ensemble
        )

        array_module = cp if HAS_CUPY else np
        assert array_module.allclose(mock_state.p, array_module.zeros((3, 3)))

        sim.step()

        # Should have initialized automatically
        # 应该自动初始化
        assert not array_module.allclose(mock_state.p, array_module.zeros((3, 3)))

    def test_step_calls_observers(self, mock_state, mock_context):
        """Test that step() calls all observers.
        测试 step() 调用所有观察者。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        observer1 = MockObserver()
        observer2 = MockObserver()

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble,
            observers=[observer1, observer2]
        )

        assert observer1.call_count == 0
        assert observer2.call_count == 0

        sim.step()

        assert observer1.call_count == 1
        assert observer2.call_count == 1

        sim.step()

        assert observer1.call_count == 2
        assert observer2.call_count == 2


class TestSimulationRun:
    """Test simulation running.
    测试模拟运行。
    """

    def test_run_multiple_steps(self, mock_state, mock_context):
        """Test that run() executes correct number of steps.
        测试 run() 执行正确数量的步数。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble
        )

        nsteps = 5
        sim.run(nsteps)

        assert sim.step_count == nsteps
        assert abs(sim.time - nsteps * 0.001) < 1e-15

    def test_run_zero_steps(self, mock_state, mock_context):
        """Test that run(0) doesn't change state.
        测试 run(0) 不改变状态。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble
        )

        sim.run(0)

        assert sim.step_count == 0
        assert sim.time == 0.0


class TestSimulationFinalize:
    """Test simulation finalization.
    测试模拟最终化。
    """

    def test_finalize_calls_observers(self, mock_state, mock_context):
        """Test that finalize() calls observer.finalize().
        测试 finalize() 调用 observer.finalize()。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        observer = MockObserver()

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble,
            observers=[observer]
        )

        assert not observer.finalized
        sim.finalize()
        assert observer.finalized

    def test_finalize_all_observers(self, mock_state, mock_context):
        """Test that finalize() calls all observer finalizers.
        测试 finalize() 调用所有观察者的终结器。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        observer1 = MockObserver()
        observer2 = MockObserver()

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[],
            ensemble=ensemble,
            observers=[observer1, observer2]
        )

        sim.finalize()

        assert observer1.finalized
        assert observer2.finalized


class TestSimulationWorkflow:
    """Test complete simulation workflows.
    测试完整的模拟工作流。
    """

    def test_full_simulation_workflow(self, mock_state, mock_context):
        """Test a complete simulation: initialize -> run -> finalize.
        测试完整的模拟：初始化 -> 运行 -> 最终化。
        """
        force_model = MockForceModel()
        ensemble = VelocityVerlet(force_model)
        observer = MockObserver()

        sim = Simulation(
            state=mock_state,
            context=mock_context,
            dt=0.001,
            initializer=[MaxwellBoltzmannDistribution(target_temp=300.0)],
            ensemble=ensemble,
            observers=[observer]
        )

        # Initialize
        # 初始化
        sim.initialize()
        assert observer.initialized

        # Run
        # 运行
        sim.run(3)
        assert sim.step_count == 3
        assert observer.call_count == 3

        # Finalize
        # 最终化
        sim.finalize()
        assert observer.finalized
