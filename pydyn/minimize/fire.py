"""FIRE (Fast Inertial Relaxation Engine) structural optimizer.
FIRE（快速惯性弛豫引擎）结构优化器。

Reference: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006)
"""

import cupy as cp


class FIRE:
    """FIRE optimizer operating on a Filter's unified DOF space.
    在 Filter 统一自由度空间上运行的 FIRE 优化器。

    Works with any Filter or CompositeFilter without modification:
    the optimizer sees only a flat position vector and a flat force vector.
    / 无需修改即可与任何 Filter 或 CompositeFilter 配合使用：
    优化器只看到平坦的位置向量和平坦的力向量。

    The internal velocity (self._v) is a fictitious velocity in DOF space
    with unit effective mass. For cell DOF and spin DOF this has no physical
    meaning beyond driving convergence.
    / 内部速度 (self._v) 是自由度空间中的虚拟速度，有效质量为单位质量。
    对于晶胞自由度和自旋自由度，这没有超出驱动收敛的物理意义。

    Example:
        # Relax atoms only
        # 仅弛豫原子
        opt = FIRE(AtomFilter(state))
        min_driver = Minimization(opt, force_model, context)
        min_driver.run(max_steps=500, fmax=0.05)

        # Relax atoms + cell + spins
        # 弛豫原子、晶胞和自旋
        opt = FIRE(CompositeFilter(state, [
            AtomFilter(state),
            CellFilter(state),
            SpinFilter(state),
        ]))
    """

    def __init__(
        self,
        filter,
        dt_start: float = 0.01,
        dt_max: float = 0.1,
        N_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        alpha_start: float = 0.1,
        f_alpha: float = 0.99,
    ):
        """
        Args:
            filter: Filter or CompositeFilter instance.
                / Filter 或 CompositeFilter 实例。
            dt_start: Initial timestep (fictitious time units).
                / 初始时间步（虚拟时间单位）。
            dt_max: Maximum allowed timestep. / 最大允许时间步。
            N_min: Steps with P > 0 before dt is increased.
                / P > 0 的步数超过此值后增大 dt。
            f_inc: Factor to increase dt. / 增大 dt 的因子。
            f_dec: Factor to decrease dt after P <= 0. / P <= 0 后减小 dt 的因子。
            alpha_start: Initial velocity mixing parameter.
                / 初始速度混合参数。
            f_alpha: Factor to decrease alpha. / 减小 alpha 的因子。
        """
        self.filter = filter
        self.dt = dt_start
        self.dt_max = dt_max
        self.N_min = N_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.f_alpha = f_alpha

        self._N_pos: int = 0
        self._v: cp.ndarray = cp.zeros(filter.n_dof())
        self._last_forces: cp.ndarray = None

    def step(self, force_model, context) -> None:
        """Perform one FIRE step.
        执行一个 FIRE 步骤。

        Args:
            force_model: Force model instance. / 力场模型实例。
            context: Simulation context. / 模拟环境。
        """
        f = self.filter.get_forces(force_model, context)
        self._last_forces = f
        x = self.filter.get_positions()

        # --- velocity update (unit-mass semi-implicit Euler) ---
        self._v += self.dt * f

        # --- FIRE power and velocity mixing ---
        P = float(cp.dot(f, self._v))
        f_norm = float(cp.linalg.norm(f))
        v_norm = float(cp.linalg.norm(self._v))

        if f_norm > 0:
            # Mix velocity toward force direction
            # 将速度混合到力的方向
            self._v = (
                (1.0 - self.alpha) * self._v
                + self.alpha * (f / f_norm) * v_norm
            )

        # --- dt and alpha adaptation ---
        if P > 0:
            self._N_pos += 1
            if self._N_pos > self.N_min:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.alpha *= self.f_alpha
        else:
            self._N_pos = 0
            self._v = cp.zeros_like(self._v)
            self.dt *= self.f_dec
            self.alpha = self.alpha_start

        # --- position update ---
        self.filter.set_positions(x + self.dt * self._v)

    def converged(self, fmax: float) -> bool:
        """Check convergence using forces from the last step.
        使用上一步的力检查收敛性。

        Returns False before any step has been taken.
        / 在执行任何步骤之前返回 False。

        Args:
            fmax: Force convergence threshold. / 力收敛阈值。

        Returns:
            True if max |force| < fmax. / 如果 max |force| < fmax 则为 True。
        """
        if self._last_forces is None:
            return False
        return bool(cp.max(cp.abs(self._last_forces)) < fmax)

    def reset(self) -> None:
        """Reset optimizer state (velocity, dt, alpha).
        重置优化器状态（速度、dt、alpha）。

        Call this if the filter's positions are externally modified between runs.
        / 如果过滤器的位置在运行之间被外部修改，请调用此方法。
        """
        self._v = cp.zeros_like(self._v)
        self._N_pos = 0
        self._last_forces = None
        self.alpha = self.alpha_start
