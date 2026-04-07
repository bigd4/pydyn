"""Plugin system tests.
插件系统测试。

Tests verify plugin registration, retrieval, listing, and error handling.
"""

import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False

from pydyn.plugins import (
    ForcePlugin, EnsemblePlugin, register_plugin,
    list_plugins, get_plugin, _force_plugins, _ensemble_plugins
)


class SimpleForcePlugin(ForcePlugin):
    """Simple test force plugin.
    简单的测试力插件。
    """
    name = "simple_force"
    version = "1.0"
    description = "A simple test force plugin"

    def compute(self, state, context, properties=None):
        """Dummy compute method.
        虚拟计算方法。
        """
        pass


class SimpleEnsemblePlugin(EnsemblePlugin):
    """Simple test ensemble plugin.
    简单的测试系综插件。
    """
    name = "simple_ensemble"
    version = "1.0"
    description = "A simple test ensemble plugin"

    def step(self, state, context, dt):
        """Dummy step method.
        虚拟步方法。
        """
        pass


@pytest.fixture(autouse=True)
def cleanup_plugins():
    """Clean up plugins after each test.
    清理每个测试后的插件。
    """
    yield
    # Clear the registries
    # 清空注册表
    _force_plugins.clear()
    _ensemble_plugins.clear()


class TestPluginRegistration:
    """Test plugin registration.
    测试插件注册。
    """

    def test_register_force_plugin(self):
        """Test registering a force plugin.
        测试注册力插件。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")

        assert "simple_force" in _force_plugins
        assert _force_plugins["simple_force"] is SimpleForcePlugin

    def test_register_ensemble_plugin(self):
        """Test registering an ensemble plugin.
        测试注册系综插件。
        """
        register_plugin(SimpleEnsemblePlugin, plugin_type="ensemble")

        assert "simple_ensemble" in _ensemble_plugins
        assert _ensemble_plugins["simple_ensemble"] is SimpleEnsemblePlugin

    def test_register_invalid_type_raises(self):
        """Test that invalid plugin_type raises ValueError.
        测试无效的 plugin_type 抛出 ValueError。
        """
        with pytest.raises(ValueError, match="Invalid plugin_type"):
            register_plugin(SimpleForcePlugin, plugin_type="invalid")

    def test_register_without_name_raises(self):
        """Test that plugin without name raises ValueError.
        测试没有名称的插件抛出 ValueError。
        """
        class BadPlugin(ForcePlugin):
            name = None

            def compute(self, state, context, properties=None):
                pass

        with pytest.raises(ValueError, match="must define a 'name'"):
            register_plugin(BadPlugin, plugin_type="force")

    def test_register_duplicate_name_raises(self):
        """Test that duplicate name raises KeyError.
        测试重复的名称抛出 KeyError。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")

        with pytest.raises(ValueError, match="already registered"):
            register_plugin(SimpleForcePlugin, plugin_type="force")

    def test_register_multiple_plugins(self):
        """Test registering multiple plugins.
        测试注册多个插件。
        """
        class ForcePluginA(ForcePlugin):
            name = "force_a"
            version = "1.0"

            def compute(self, state, context, properties=None):
                pass

        class ForcePluginB(ForcePlugin):
            name = "force_b"
            version = "1.0"

            def compute(self, state, context, properties=None):
                pass

        register_plugin(ForcePluginA, plugin_type="force")
        register_plugin(ForcePluginB, plugin_type="force")

        assert "force_a" in _force_plugins
        assert "force_b" in _force_plugins


class TestPluginListing:
    """Test plugin listing.
    测试插件列表。
    """

    def test_list_plugins_empty(self):
        """Test listing plugins when none are registered.
        测试未注册插件时列出插件。
        """
        plugins = list_plugins()
        assert isinstance(plugins, dict)
        assert plugins.get("force", {}) == {}
        assert plugins.get("ensemble", {}) == {}

    def test_list_force_plugins(self):
        """Test listing force plugins.
        测试列出力插件。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")

        plugins = list_plugins(plugin_type="force")

        assert "force" in plugins
        assert "simple_force" in plugins["force"]
        assert plugins["force"]["simple_force"]["version"] == "1.0"

    def test_list_ensemble_plugins(self):
        """Test listing ensemble plugins.
        测试列出系综插件。
        """
        register_plugin(SimpleEnsemblePlugin, plugin_type="ensemble")

        plugins = list_plugins(plugin_type="ensemble")

        assert "ensemble" in plugins
        assert "simple_ensemble" in plugins["ensemble"]
        assert plugins["ensemble"]["simple_ensemble"]["version"] == "1.0"

    def test_list_all_plugins(self):
        """Test listing all plugins.
        测试列出所有插件。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")
        register_plugin(SimpleEnsemblePlugin, plugin_type="ensemble")

        plugins = list_plugins(plugin_type="all")

        assert "force" in plugins
        assert "ensemble" in plugins
        assert "simple_force" in plugins["force"]
        assert "simple_ensemble" in plugins["ensemble"]

    def test_list_plugins_invalid_type_raises(self):
        """Test that invalid plugin_type raises ValueError.
        测试无效的 plugin_type 抛出 ValueError。
        """
        with pytest.raises(ValueError, match="Invalid plugin_type"):
            list_plugins(plugin_type="invalid")

    def test_list_plugins_metadata(self):
        """Test that plugin metadata is included in listing.
        测试插件元数据包含在列表中。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")

        plugins = list_plugins()

        entry = plugins["force"]["simple_force"]
        assert "version" in entry
        assert "description" in entry
        assert entry["version"] == "1.0"
        assert entry["description"] == "A simple test force plugin"


class TestPluginRetrieval:
    """Test plugin retrieval.
    测试插件检索。
    """

    def test_get_registered_force_plugin(self):
        """Test getting a registered force plugin.
        测试获取已注册的力插件。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")

        plugin = get_plugin("simple_force", plugin_type="force")

        assert plugin is SimpleForcePlugin

    def test_get_registered_ensemble_plugin(self):
        """Test getting a registered ensemble plugin.
        测试获取已注册的系综插件。
        """
        register_plugin(SimpleEnsemblePlugin, plugin_type="ensemble")

        plugin = get_plugin("simple_ensemble", plugin_type="ensemble")

        assert plugin is SimpleEnsemblePlugin

    def test_get_nonexistent_plugin_returns_none(self):
        """Test that getting nonexistent plugin returns None.
        测试获取不存在的插件返回 None。
        """
        plugin = get_plugin("nonexistent", plugin_type="force")

        assert plugin is None

    def test_get_plugin_wrong_type_returns_none(self):
        """Test that getting plugin with wrong type returns None.
        测试使用错误类型获取插件返回 None。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")

        plugin = get_plugin("simple_force", plugin_type="ensemble")

        assert plugin is None


class TestPluginValidation:
    """Test plugin validation.
    测试插件验证。
    """

    def test_plugin_validate_method_default(self):
        """Test that default validate returns True for valid plugin.
        测试默认验证对有效插件返回 True。
        """
        plugin = SimpleForcePlugin()
        assert plugin.validate() is True

    def test_plugin_without_name_fails_validation(self):
        """Test that plugin without name fails validation.
        测试没有名称的插件验证失败。
        """
        class BadPlugin(ForcePlugin):
            name = None

            def compute(self, state, context, properties=None):
                pass

        plugin = BadPlugin()
        assert plugin.validate() is False

    def test_custom_validation(self):
        """Test custom validation in plugin.
        测试插件中的自定义验证。
        """
        class ValidatedPlugin(ForcePlugin):
            name = "validated"
            version = "1.0"

            def validate(self):
                return self.name == "validated"

            def compute(self, state, context, properties=None):
                pass

        plugin = ValidatedPlugin()
        assert plugin.validate() is True


class TestForcePluginInterface:
    """Test ForcePlugin interface.
    测试 ForcePlugin 接口。
    """

    def test_force_plugin_abstract_compute(self):
        """Test that ForcePlugin compute is abstract.
        测试 ForcePlugin compute 是抽象的。
        """
        # Cannot instantiate abstract class with unimplemented abstract method
        # 无法实例化具有未实现的抽象方法的抽象类
        with pytest.raises(TypeError, match="abstract"):
            plugin = ForcePlugin()

    def test_force_plugin_default_attributes(self):
        """Test ForcePlugin default attributes.
        测试 ForcePlugin 默认属性。
        """
        plugin = SimpleForcePlugin()

        assert plugin.name == "simple_force"
        assert plugin.version == "1.0"
        assert isinstance(plugin.description, str)


class TestEnsemblePluginInterface:
    """Test EnsemblePlugin interface.
    测试 EnsemblePlugin 接口。
    """

    def test_ensemble_plugin_abstract_step(self):
        """Test that EnsemblePlugin step is abstract.
        测试 EnsemblePlugin step 是抽象的。
        """
        # Cannot instantiate abstract class with unimplemented abstract method
        # 无法实例化具有未实现的抽象方法的抽象类
        with pytest.raises(TypeError, match="abstract"):
            plugin = EnsemblePlugin()

    def test_ensemble_plugin_default_methods(self):
        """Test EnsemblePlugin default optional methods.
        测试 EnsemblePlugin 默认可选方法。
        """
        plugin = SimpleEnsemblePlugin()

        assert plugin.get_conserved_energy(None, None) is None
        assert plugin.get_pressure(None, None) is None

    def test_ensemble_plugin_custom_conserved_energy(self):
        """Test EnsemblePlugin can provide conserved energy.
        测试 EnsemblePlugin 可以提供守恒能量。
        """
        class EnsembleWithEnergy(EnsemblePlugin):
            name = "with_energy"
            version = "1.0"

            def step(self, state, context, dt):
                pass

            def get_conserved_energy(self, state, context):
                return 42.0

        plugin = EnsembleWithEnergy()
        energy = plugin.get_conserved_energy(None, None)

        assert energy == 42.0


class TestPluginWorkflow:
    """Test complete plugin workflows.
    测试完整的插件工作流。
    """

    def test_register_and_retrieve_workflow(self):
        """Test registering and retrieving a plugin.
        测试注册和检索插件。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")

        # Retrieve via get_plugin
        # 通过 get_plugin 检索
        plugin_class = get_plugin("simple_force", plugin_type="force")
        assert plugin_class is SimpleForcePlugin

        # Also appears in list
        # 也出现在列表中
        plugins = list_plugins()
        assert "simple_force" in plugins["force"]

    def test_multiple_plugin_types_workflow(self):
        """Test handling multiple plugin types.
        测试处理多个插件类型。
        """
        register_plugin(SimpleForcePlugin, plugin_type="force")
        register_plugin(SimpleEnsemblePlugin, plugin_type="ensemble")

        force = get_plugin("simple_force", plugin_type="force")
        ensemble = get_plugin("simple_ensemble", plugin_type="ensemble")

        assert force is SimpleForcePlugin
        assert ensemble is SimpleEnsemblePlugin
        assert force != ensemble
