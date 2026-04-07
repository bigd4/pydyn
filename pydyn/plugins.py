"""Plugin and extension system for PyDyn extensibility.
PyDyn 可扩展性的插件和扩展系统。

This module provides base classes and utilities for extending PyDyn with
custom force models and ensemble integrators through a plugin architecture.
/ 此模块为通过插件架构使用自定义力模型和系综积分器
扩展 PyDyn 提供基类和实用程序。
"""
from typing import Dict, Type, Optional, Any
from abc import ABC, abstractmethod


# Global plugin registry
# 全局插件注册表
_force_plugins: Dict[str, Type["ForcePlugin"]] = {}
_ensemble_plugins: Dict[str, Type["EnsemblePlugin"]] = {}


class ForcePlugin(ABC):
    """Base class for custom force model plugins.
    自定义力模型插件的基类。

    Subclass this to create custom force models that can be registered and
    used within PyDyn simulations. Force plugins encapsulate potential
    energy calculations, atomic forces, and related properties.
    / 子类化以创建可在 PyDyn 模拟中注册和使用的自定义力模型。
    力插件封装势能计算、原子力和相关属性。

    Example:
        class MyCustomForceModel(ForcePlugin):
            name = "custom_force"
            version = "1.0"

            def compute(self, state, context, properties=None):
                # Implementation of force calculation
                pass
    """

    name: str = None  # Must be overridden in subclass
    # 必须在子类中覆盖
    version: str = "1.0"  # Plugin version
    # 插件版本
    description: str = ""  # Human-readable description
    # 人类可读的描述

    @abstractmethod
    def compute(self, state: Any, context: Any, properties: Optional[list] = None) -> None:
        """Compute forces and properties for the system.
        计算系统的力和属性。

        This method must be implemented to compute forces, potential energy,
        virials, or other properties as specified in the properties list.
        / 必须实现此方法以计算力、势能、维里或 properties
        列表中指定的其他属性。

        Args:
            state: Current simulation state with positions, momenta, etc.
                / 当前模拟状态（包含位置、动量等）。
            context: Simulation context with cell, temperature, etc.
                / 模拟环境（包含盒、温度等）。
            properties: Optional list of properties to compute.
                If None, compute all available properties.
                / 要计算的可选属性列表。如果为 None，则计算所有可用属性。
        """
        pass

    def validate(self) -> bool:
        """Validate plugin configuration and compatibility.
        验证插件配置和兼容性。

        Override this method to perform custom validation. Return False
        if the plugin cannot be used in the current environment.
        / 重写此方法以执行自定义验证。如果插件不能在当前环境中使用，
        则返回 False。

        Returns:
            True if plugin is valid and usable. / 如果插件有效且可用则返回 True。
        """
        return self.name is not None


class EnsemblePlugin(ABC):
    """Base class for custom ensemble integrator plugins.
    自定义系综积分器插件的基类。

    Subclass this to create custom ensemble integrators (e.g., NVT, NPT,
    spin systems) that can be registered and used within PyDyn.
    / 子类化以创建可在 PyDyn 中注册和使用的自定义系综积分器
    （例如 NVT、NPT、自旋系统）。

    Example:
        class MyCustomEnsemble(EnsemblePlugin):
            name = "custom_ensemble"
            version = "1.0"

            def __init__(self, force_model, parameters):
                self.force_model = force_model
                # Initialize with parameters

            def step(self, state, context, dt):
                # Implementation of integration step
                pass
    """

    name: str = None  # Must be overridden in subclass
    # 必须在子类中覆盖
    version: str = "1.0"  # Plugin version
    # 插件版本
    description: str = ""  # Human-readable description
    # 人类可读的描述

    @abstractmethod
    def step(self, state: Any, context: Any, dt: float) -> None:
        """Execute one integration step.
        执行一个积分步。

        This method must be implemented to perform one timestep of integration
        using the specific ensemble's dynamics and integration scheme.
        / 必须实现此方法以使用特定系综的动力学和积分方案
        执行一个积分时间步。

        Args:
            state: Current simulation state. / 当前模拟状态。
            context: Simulation context. / 模拟环境。
            dt: Timestep duration in picoseconds. / 时间步长（皮秒）。
        """
        pass

    def get_conserved_energy(self, state: Any, context: Any) -> Optional[float]:
        """Get the conserved energy quantity for this ensemble (optional).
        获取此系综的守恒能量量（可选）。

        Override this method if your ensemble maintains a conserved quantity
        (e.g., modified Hamiltonian in Nosé-Hoover).
        / 如果您的系综维持守恒量（例如 Nosé-Hoover 中的修改哈密顿量），
        请覆盖此方法。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context. / 模拟环境。

        Returns:
            Conserved energy, or None if not applicable.
            / 守恒能量，或如果不适用则返回 None。
        """
        return None

    def get_pressure(self, state: Any, context: Any) -> Optional[float]:
        """Get the system pressure (optional).
        获取系统压力（可选）。

        Override this method if your ensemble calculates pressure
        (e.g., NPT ensembles).
        / 如果您的系综计算压力（例如 NPT 系综），请覆盖此方法。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context. / 模拟环境。

        Returns:
            Pressure in Bar or other units, or None if not applicable.
            / Bar 或其他单位的压力，或如果不适用则返回 None。
        """
        return None

    def validate(self) -> bool:
        """Validate plugin configuration and compatibility.
        验证插件配置和兼容性。

        Override this method to perform custom validation. Return False
        if the plugin cannot be used in the current environment.
        / 重写此方法以执行自定义验证。如果插件不能在当前环境中使用，
        则返回 False。

        Returns:
            True if plugin is valid and usable. / 如果插件有效且可用则返回 True。
        """
        return self.name is not None


def register_plugin(
    plugin_class: Type[Any],
    plugin_type: str = "force"
) -> None:
    """Register a force or ensemble plugin.
    注册力或系综插件。

    Plugins must have a unique 'name' attribute and implement the required
    abstract methods. Call this function to register a plugin before using it.
    / 插件必须具有唯一的 'name' 属性并实现所需的抽象方法。
    在使用插件之前调用此函数以注册插件。

    Args:
        plugin_class: The plugin class to register.
            / 要注册的插件类。
        plugin_type: Type of plugin: "force" or "ensemble".
            / 插件类型："force" 或 "ensemble"。

    Raises:
        ValueError: If plugin_type is invalid or plugin lacks required attributes.
            / 如果 plugin_type 无效或插件缺少必需属性，则引发 ValueError。
        KeyError: If a plugin with the same name is already registered.
            / 如果已注册相同名称的插件，则引发 KeyError。
    """
    if plugin_type not in ("force", "ensemble"):
        raise ValueError(f"Invalid plugin_type '{plugin_type}'. Must be 'force' or 'ensemble'.")

    if not hasattr(plugin_class, "name") or plugin_class.name is None:
        raise ValueError(f"Plugin class {plugin_class.__name__} must define a 'name' attribute.")

    # Instantiate plugin to validate
    # 实例化插件以验证
    try:
        if plugin_type == "force":
            # Force plugins may require arguments to __init__
            # 力插件的 __init__ 可能需要参数
            if plugin_class.name in _force_plugins:
                raise KeyError(
                    f"Force plugin '{plugin_class.name}' is already registered."
                )
            _force_plugins[plugin_class.name] = plugin_class
        else:
            # Ensemble plugins may require arguments to __init__
            # 系综插件的 __init__ 可能需要参数
            if plugin_class.name in _ensemble_plugins:
                raise KeyError(
                    f"Ensemble plugin '{plugin_class.name}' is already registered."
                )
            _ensemble_plugins[plugin_class.name] = plugin_class
    except Exception as e:
        raise ValueError(f"Failed to register plugin: {str(e)}")


def list_plugins(plugin_type: str = "all") -> Dict[str, Dict[str, str]]:
    """List all registered plugins.
    列出所有已注册的插件。

    Args:
        plugin_type: Type of plugins to list: "force", "ensemble", or "all".
            / 要列出的插件类型："force"、"ensemble" 或 "all"。

    Returns:
        Dictionary mapping plugin names to metadata (name, version, description).
        / 将插件名称映射到元数据（名称、版本、描述）的字典。

    Raises:
        ValueError: If plugin_type is invalid.
            / 如果 plugin_type 无效，则引发 ValueError。
    """
    if plugin_type not in ("force", "ensemble", "all"):
        raise ValueError(
            f"Invalid plugin_type '{plugin_type}'. Must be 'force', 'ensemble', or 'all'."
        )

    result = {}

    if plugin_type in ("force", "all"):
        result["force"] = {
            name: {
                "version": plugin_class.version,
                "description": plugin_class.description,
            }
            for name, plugin_class in _force_plugins.items()
        }

    if plugin_type in ("ensemble", "all"):
        result["ensemble"] = {
            name: {
                "version": plugin_class.version,
                "description": plugin_class.description,
            }
            for name, plugin_class in _ensemble_plugins.items()
        }

    return result


def get_plugin(name: str, plugin_type: str = "force") -> Optional[Type[Any]]:
    """Retrieve a registered plugin by name.
    按名称检索已注册的插件。

    Args:
        name: Name of the plugin. / 插件的名称。
        plugin_type: Type of plugin: "force" or "ensemble".
            / 插件类型："force" 或 "ensemble"。

    Returns:
        The plugin class, or None if not found.
        / 插件类，或如果未找到则返回 None。
    """
    if plugin_type == "force":
        return _force_plugins.get(name)
    elif plugin_type == "ensemble":
        return _ensemble_plugins.get(name)
    return None
