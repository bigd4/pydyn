"""Force model tests.
力模型测试。

Tests verify force model interface, result caching, and compute behavior.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from pydyn.forces.base import ForceModel
from pydyn.state import State
from pydyn.context import SimulationContext


class SimpleForceModel(ForceModel):
    """Simple test force model.
    简单的测试力模型。
    """
    implemented_properties = ["forces", "energy"]

    def compute(self, state, context, properties=None):
        """Return dummy forces and energy.
        返回虚拟力和能量。
        """
        array_module = cp if HAS_CUPY else np
        self.results["forces"] = array_module.ones_like(state.r) * 0.1
        self.results["energy"] = 10.0
        self.state = state.copy()


class TestForceModelInterface:
    """Test ForceModel base class interface.
    测试 ForceModel 基类接口。
    """

    def test_forcemodel_initialization(self):
        """Test that ForceModel can be instantiated.
        测试可以实例化 ForceModel。
        """
        model = SimpleForceModel()
        assert model is not None
        assert hasattr(model, 'results')
        assert hasattr(model, 'state')

    def test_results_dict_empty_initially(self):
        """Test that results dictionary is empty initially.
        测试结果字典最初为空。
        """
        model = SimpleForceModel()
        assert isinstance(model.results, dict)
        assert len(model.results) == 0

    def test_state_none_initially(self):
        """Test that cached state is None initially.
        测试缓存状态最初为 None。
        """
        model = SimpleForceModel()
        assert model.state is None


class TestForceModelCompute:
    """Test force computation.
    测试力计算。
    """

    def test_compute_executes(self, mock_state):
        """Test that compute() can be executed.
        测试 compute() 可以执行。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        # Should not raise
        # 不应该抛出
        model.compute(mock_state, context)

    def test_compute_fills_results(self, mock_state):
        """Test that compute fills results dictionary.
        测试 compute 填充结果字典。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        model.compute(mock_state, context)

        assert "forces" in model.results
        assert "energy" in model.results

    def test_compute_with_properties_list(self, mock_state):
        """Test compute with explicit properties list.
        测试用显式属性列表计算。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        model.compute(mock_state, context, properties=["forces"])

        assert "forces" in model.results

    def test_compute_with_none_properties(self, mock_state):
        """Test compute with None properties (compute all).
        测试计算 None 属性（计算全部）。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        model.compute(mock_state, context, properties=None)

        # Should compute all available properties
        # 应该计算所有可用属性
        assert len(model.results) > 0


class TestForceModelResultCaching:
    """Test result caching behavior.
    测试结果缓存行为。
    """

    def test_need_compute_initial_true(self, mock_state):
        """Test that need_compute returns True initially.
        测试 need_compute 最初返回 True。
        """
        model = SimpleForceModel()

        result = model.need_compute(mock_state, None, ["forces"])
        assert result is True

    def test_need_compute_same_state_false(self, mock_state):
        """Test that need_compute returns False for same state.
        测试 need_compute 对相同状态返回 False。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        # First compute
        # 第一次计算
        model.compute(mock_state, context, properties=["forces"])

        # Second call with same state should return False
        # 对相同状态的第二次调用应返回 False
        result = model.need_compute(mock_state, context, ["forces"])
        assert result is False

    def test_need_compute_different_state_true(self, mock_state):
        """Test that need_compute returns True for different state.
        测试 need_compute 对不同状态返回 True。
        """
        array_module = cp if HAS_CUPY else np

        model = SimpleForceModel()
        context = SimulationContext()

        # First compute
        # 第一次计算
        model.compute(mock_state, context, properties=["forces"])

        # Create different state
        # 创建不同的状态
        state2 = State(
            r=array_module.array([[1.0, 0.0, 0.0]], dtype=float),
            p=array_module.zeros((1, 3), dtype=float),
            m=array_module.array([1.0], dtype=float),
            box=array_module.eye(3, dtype=float) * 10,
            atomic_number=array_module.array([18], dtype=int),
            pbc=array_module.array([True, True, True])
        )

        # Second call with different state should return True
        # 对不同状态的第二次调用应返回 True
        result = model.need_compute(state2, context, ["forces"])
        assert result is True

    def test_need_compute_missing_property_true(self, mock_state):
        """Test that need_compute returns True if property is missing.
        测试如果缺少属性，need_compute 返回 True。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        # First compute with both properties
        # 首先用两个属性计算
        model.compute(mock_state, context, properties=["forces", "energy"])

        # Now clear the results but keep the state cached
        # 现在清空结果但保留缓存的状态
        model.results.clear()

        # Request both properties again - should need compute because results are empty
        # 再次请求两个属性 - 应该需要计算因为结果为空
        result = model.need_compute(mock_state, context, ["forces", "energy"])
        assert result is True


class TestForceModelGetResult:
    """Test result retrieval.
    测试结果检索。
    """

    def test_get_result_existing_property(self, mock_state):
        """Test retrieving existing property.
        测试检索现有属性。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        model.compute(mock_state, context)

        forces = model.get_result("forces")
        assert forces is not None

    def test_get_result_missing_property_raises(self, mock_state):
        """Test that missing property raises KeyError.
        测试缺失的属性会引发 KeyError。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        model.compute(mock_state, context, properties=["forces"])

        with pytest.raises(KeyError):
            model.get_result("nonexistent")

    def test_get_result_before_compute_raises(self):
        """Test that get_result before compute raises KeyError.
        测试在计算之前调用 get_result 会引发 KeyError。
        """
        model = SimpleForceModel()

        with pytest.raises(KeyError):
            model.get_result("forces")

    def test_get_result_error_message_informative(self, mock_state):
        """Test that error message lists available properties.
        测试错误消息列出可用的属性。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        model.compute(mock_state, context)

        try:
            model.get_result("missing")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            error_msg = str(e)
            # Error message should mention available properties
            # 错误消息应该提及可用的属性
            assert "Available properties" in error_msg


class TestForceModelProperties:
    """Test force model properties.
    测试力模型属性。
    """

    def test_implemented_properties(self):
        """Test that implemented_properties is defined.
        测试 implemented_properties 已定义。
        """
        model = SimpleForceModel()
        assert hasattr(model, 'implemented_properties')
        assert isinstance(model.implemented_properties, list)

    def test_available_properties_empty_initially(self):
        """Test available_properties is empty initially.
        测试 available_properties 最初为空。
        """
        model = SimpleForceModel()
        assert len(model.available_properties) == 0

    def test_available_properties_after_compute(self, mock_state):
        """Test available_properties reflects computed properties.
        测试 available_properties 反映计算的属性。
        """
        model = SimpleForceModel()
        context = SimulationContext()

        model.compute(mock_state, context)

        props = model.available_properties
        assert "forces" in props
        assert "energy" in props


class TestAbstractForceModel:
    """Test abstract ForceModel behavior.
    测试抽象 ForceModel 行为。
    """

    def test_abstract_compute_not_implemented(self):
        """Test that abstract compute raises NotImplementedError.
        测试抽象 compute 抛出 NotImplementedError。
        """
        model = ForceModel()

        with pytest.raises(NotImplementedError):
            model.compute(None, None)

    def test_force_model_can_be_subclassed(self):
        """Test that ForceModel can be subclassed.
        测试 ForceModel 可以被子类化。
        """
        class CustomForce(ForceModel):
            def compute(self, state, context, properties=None):
                array_module = cp if HAS_CUPY else np
                self.results["custom"] = 42.0

        model = CustomForce()
        assert isinstance(model, ForceModel)
