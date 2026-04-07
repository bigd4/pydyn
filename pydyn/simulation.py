"""Molecular dynamics simulation driver.
分子动力学模拟驱动程序。

This module provides the main Simulation class for running MD simulations
with configurable ensembles, initializers, and observers.
"""

from typing import List, Optional, Any


class Simulation:
    """Main simulation driver for molecular dynamics.
    分子动力学模拟的主驱动程序。

    Coordinates the simulation state, ensemble propagation, initialization,
    constraints, and observation throughout a trajectory.

    Attributes:
        state: Simulation state containing atomic positions, momenta, etc.
            / 包含原子位置、动量等的模拟状态。
        context: Simulation context with thermodynamic parameters.
            / 包含热力学参数的模拟环境。
        ensemble: Ensemble (NVE, NVT, NPT, etc.) for propagation.
            / 用于传播的系综（NVE、NVT、NPT 等）。
        dt: Timestep in picoseconds. / 时间步长（皮秒）。
        initializer: List of velocity/configuration initializers.
            / 速度/配置初始化程序的列表。
        observers: List of trajectory observers. / 轨迹观察者列表。
        time: Elapsed simulation time in picoseconds. / 已运行的模拟时间（皮秒）。
        step_count: Number of simulation steps completed.
            / 已完成的模拟步数。
    """

    def __init__(
        self,
        state: Any,
        context: Any,
        dt: float,
        initializer: List[Any],
        ensemble: Any,
        observers: Optional[List[Any]] = None,
    ) -> None:
        """Initialize a molecular dynamics simulation.
        初始化分子动力学模拟。

        Args:
            state: State object with atomic positions and momenta.
                / 包含原子位置和动量的状态对象。
            context: SimulationContext with target thermodynamic conditions.
                / 带有目标热力学条件的 SimulationContext。
            dt: Timestep in picoseconds. / 时间步长（皮秒）。
            initializer: List of initializer objects. / 初始化程序对象列表。
            ensemble: Ensemble object for propagating dynamics.
                / 用于传播动力学的系综对象。
            observers: Optional list of observer objects to track trajectory.
                / 用于跟踪轨迹的观察者对象的可选列表。
        """
        self.state = state
        self.ensemble = ensemble
        self.context = context
        self.dt = dt
        self.initializer = initializer

        self.time: float = 0.0
        self.step_count: int = 0
        self.observers: List[Any] = observers or []

        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize the simulation.
        初始化模拟。

        Runs all initializers, extends state with ensemble-specific variables,
        applies constraints, and initializes observers. Can be called multiple
        times safely (subsequent calls are no-ops).
        / 运行所有初始化程序，用系综特定变量扩展状态，应用约束并初始化观察者。
        可以安全地多次调用（后续调用为无操作）。
        """
        if self._initialized:
            return

        for init in self.initializer:
            init.initialize(self.state, self.context)

        for op, _ in self.ensemble.op_list:
            if hasattr(op, "extend_state"):
                op.extend_state(self.state, self.context)

        for constraint in self.context.constraints:
            constraint.apply(self.state, self.context)

        for obs in self.observers:
            obs.initialize()

        self._initialized = True

    def step(self) -> None:
        """Advance the simulation by one timestep.
        将模拟向前推进一个时间步。

        Calls ensemble.step() to propagate dynamics, updates internal time
        and step counter, and notifies all observers.
        / 调用 ensemble.step() 传播动力学，更新内部时间和步数计数器，
        并通知所有观察者。
        """
        if not self._initialized:
            self.initialize()

        self.ensemble.step(self.state, self.context, self.dt)

        self.time += self.dt
        self.step_count += 1

        for obs in self.observers:
            obs(self)

    def run(self, nsteps: int) -> None:
        """Run the simulation for a specified number of steps.
        运行模拟指定的步数。

        Args:
            nsteps: Number of timesteps to simulate. / 要模拟的时间步数。
        """
        for _ in range(nsteps):
            self.step()

    def finalize(self) -> None:
        """Finalize the simulation.
        最终确定模拟。

        Calls finalize() on all observers to perform cleanup and final output.
        / 调用所有观察者的 finalize()，以执行清理和最终输出。
        """
        for obs in self.observers:
            obs.finalize()
