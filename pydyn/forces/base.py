"""Base force model for molecular dynamics simulations.
分子动力学模拟的基础力模型。
"""
import cupy as cp
from typing import List, Optional, Dict, Any


class ForceModel:
    """Abstract base class for force models computing atomic interactions.
    计算原子相互作用的力模型的抽象基类。

    Defines the interface for force computation and result caching. Subclasses
    should implement the compute() method to provide specific force calculations.
    / 定义力计算和结果缓存的接口。子类应实现 compute()
    方法以提供特定的力计算。

    Attributes:
        implemented_properties: List of properties this model can compute.
            / 此模型可以计算的属性列表。
        state: Cached state to avoid redundant computations.
            / 缓存的状态以避免冗余计算。
        results: Dictionary of computed properties.
            / 计算属性的字典。
    """

    implemented_properties: List[str] = []

    def __init__(self):
        """Initialize force model.
        初始化力模型。
        """
        self.state = None
        self.results: Dict[str, Any] = {}

    def need_compute(self, state, context, properties):
        """Check if computation is needed based on state and cached properties.
        根据状态和缓存属性检查是否需要计算。

        Args:
            state: Current simulation state. / 当前模拟状态。
            context: Simulation context. / 模拟环境。
            properties: List of properties to compute. / 要计算的属性列表。

        Returns:
            True if computation is needed, False if cached results are valid.
            / 如果需要计算则返回 True，如果缓存结果有效则返回 False。
        """
        if state.configure_same_as(self.state):
            for prop in properties:
                if prop not in self.results:
                    return True
            return False
        self.state = state.copy()
        self.results = {}
        return True

    def compute(self, state, context, properties: List[str] = None):
        """Compute forces and other properties for the system.
        计算系统的力和其他属性。

        This method must be implemented by subclasses to compute forces,
        energies, virials, and other properties as requested.
        / 此方法必须由子类实现以计算所需的力、能量、维里和其他属性。

        Args:
            state: Current simulation state. / 当前模拟状态。
            context: Simulation context. / 模拟环境。
            properties: List of properties to compute.
                / 要计算的属性列表。

        Raises:
            NotImplementedError: If not overridden by subclass.
                / 如果未被子类重写，则引发 NotImplementedError。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute() method"
        )

    def get_result(self, key: str) -> Any:
        """Retrieve a computed property result.
        检索计算的属性结果。

        Args:
            key: Property name to retrieve. / 要检索的属性名。

        Returns:
            The computed property value. / 计算的属性值。

        Raises:
            KeyError: If property was not computed or does not exist.
                / 如果属性未被计算或不存在，则引发 KeyError。
        """
        if key not in self.results:
            available = list(self.results.keys())
            raise KeyError(
                f"Property '{key}' not found in results. "
                f"Available properties: {available}"
            )
        return self.results[key]

    @property
    def available_properties(self) -> List[str]:
        """Get list of properties available from last computation.
        获取上次计算中可用的属性列表。

        Returns:
            List of computed property names. / 计算属性名称列表。
        """
        return list(self.results.keys())

