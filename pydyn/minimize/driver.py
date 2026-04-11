"""Minimization driver coordinating optimizer, force model, and observers.
协调优化器、力场模型和观察者的能量最小化驱动程序。
"""

from typing import List, Optional, Any


class Minimization:
    """Driver for structural optimization.
    结构优化驱动程序。

    Analogous to Simulation for dynamics: it owns the run loop, step counter,
    and observer notification. The optimizer (e.g. FIRE) owns the algorithm
    state; the filter owns the DOF mapping.
    / 类似于动力学的 Simulation：它拥有运行循环、步数计数器和观察者通知。
    优化器（如 FIRE）拥有算法状态；过滤器拥有自由度映射。

    Example:
        filter = CompositeFilter(state, [AtomFilter(state), SpinFilter(state)])
        opt = FIRE(filter)
        min_driver = Minimization(opt, force_model, context)
        n_steps = min_driver.run(max_steps=1000, fmax=0.05)

    Attributes:
        optimizer: Optimizer instance (e.g. FIRE). / 优化器实例（如 FIRE）。
        force_model: Force model. / 力场模型。
        context: Simulation context. / 模拟环境。
        observers: List of observer callables receiving this Minimization.
            / 接收此 Minimization 的观察者可调用列表。
        step_count: Number of steps taken so far. / 迄今为止执行的步数。
    """

    def __init__(
        self,
        optimizer: Any,
        force_model: Any,
        context: Any,
        observers: Optional[List[Any]] = None,
    ) -> None:
        """
        Args:
            optimizer: Optimizer with step(force_model, context) and
                converged(fmax) methods (e.g. FIRE).
                / 具有 step(force_model, context) 和 converged(fmax) 方法的优化器
                / （如 FIRE）。
            force_model: Force model instance. / 力场模型实例。
            context: Simulation context. / 模拟环境。
            observers: Optional list of observer objects (compatible with
                pydyn.observers.Observer) invoked after each step.
                / 每步之后调用的观察者对象列表（兼容 pydyn.observers.Observer）。
        """
        self.optimizer = optimizer
        self.force_model = force_model
        self.context = context
        self.observers: List[Any] = observers or []
        self.step_count: int = 0

    @property
    def state(self):
        """Current simulation state, forwarded from the filter.
        当前模拟状态，从过滤器转发。

        Exposes state so that observers designed for Simulation
        (e.g. AtomsDump, LogThermol) work without modification.
        / 暴露状态，使为 Simulation 设计的观察者（如 AtomsDump、LogThermol）
        无需修改即可使用。
        """
        return self.optimizer.filter.state

    def step(self) -> None:
        """Advance the optimization by one step and notify observers.
        将优化推进一步并通知观察者。
        """
        self.optimizer.step(self.force_model, self.context)
        self.step_count += 1
        for obs in self.observers:
            obs(self)

    def run(self, max_steps: int, fmax: float = 0.05) -> int:
        """Run until convergence or max_steps is reached.
        运行直到收敛或达到 max_steps。

        Calls initialize() on observers before the loop and finalize()
        after, matching the Simulation interface so that observers like
        AtomsDump and LogThermol work without modification.
        / 在循环前调用观察者的 initialize()，循环后调用 finalize()，
        与 Simulation 接口匹配，使 AtomsDump、LogThermol 等观察者无需修改即可使用。

        Convergence is checked after each step using forces already computed
        by that step (no extra force evaluation per step).
        / 收敛性在每步之后使用该步已计算的力进行检查
        / （每步无需额外的力评估）。

        Args:
            max_steps: Maximum number of optimizer steps.
                / 最大优化步数。
            fmax: Convergence threshold: max |force component| < fmax.
                / 收敛阈值：max |力分量| < fmax。

        Returns:
            Total number of steps taken (including steps before this call).
            / 执行的步骤总数（包括此调用之前的步骤）。
        """
        for obs in self.observers:
            if hasattr(obs, "initialize"):
                obs.initialize()
        try:
            for _ in range(max_steps):
                self.step()
                if self.optimizer.converged(fmax):
                    break
        finally:
            for obs in self.observers:
                if hasattr(obs, "finalize"):
                    obs.finalize()
        return self.step_count
