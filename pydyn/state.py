"""Simulation state representation.
模拟状态表示。

This module defines the State class and extension classes for storing
atomic positions, momenta, masses, and auxiliary properties like spins
and thermostat/barostat variables.
"""

from typing import Optional, Dict, Any
import cupy as cp
from .constants import Constants
from ase import Atoms


class State:
    """Complete simulation state for a molecular system.
    分子系统的完整模拟状态。

    Stores atomic positions, momenta, masses, cell parameters, and
    extensible components (spins, thermostat variables, barostat variables).
    All array data is stored on GPU using CuPy.

    Attributes:
        N: Number of atoms. / 原子数。
        r: Atomic positions in Angstroms, shape (N, 3).
            / 原子位置（埃），形状 (N, 3)。
        p: Atomic momenta in amu*Ang/ps, shape (N, 3).
            / 原子动量（amu*Ang/ps），形状 (N, 3)。
        m: Atomic masses in amu, shape (N,). / 原子质量（amu），形状 (N,)。
        box: Simulation cell vectors in Angstroms, shape (3, 3).
            / 模拟盒矢量（埃），形状 (3, 3)。
        atomic_number: Atomic numbers for each atom, shape (N,).
            / 每个原子的原子序数，形状 (N,)。
        pbc: Periodic boundary conditions, shape (3,).
            / 周期性边界条件，形状 (3,)。
        components: Dictionary of extension components.
            / 扩展组件的字典。
        _extensions: Internal registry for tracking registered extensions.
            / 用于跟踪已注册扩展的内部注册表。
    """

    def __init__(
        self,
        r: Optional[cp.ndarray] = None,
        p: Optional[cp.ndarray] = None,
        m: Optional[cp.ndarray] = None,
        box: Optional[cp.ndarray] = None,
        atomic_number: Optional[cp.ndarray] = None,
        pbc: Optional[cp.ndarray] = None,
        extensions: Optional[list] = None,
    ) -> None:
        """Initialize a simulation state.
        初始化模拟状态。

        Args:
            r: Atomic positions in Angstroms. / 原子位置（埃）。
            p: Atomic momenta in amu*Ang/ps. / 原子动量（amu*Ang/ps）。
            m: Atomic masses in amu. / 原子质量（amu）。
            box: Simulation cell vectors (3x3 matrix). / 模拟盒矢量（3x3矩阵）。
            atomic_number: Atomic numbers. / 原子序数。
            pbc: Periodic boundary conditions. / 周期性边界条件。
            extensions: Optional list of Extension objects.
                / 扩展对象的可选列表。
        """
        self.N: int = len(r) if r is not None else 0
        self.r: Optional[cp.ndarray] = r
        self.p: Optional[cp.ndarray] = p
        self.m: Optional[cp.ndarray] = m
        self.box: Optional[cp.ndarray] = box
        self.atomic_number: Optional[cp.ndarray] = atomic_number
        self.pbc: Optional[cp.ndarray] = pbc
        self.components: Dict[str, Any] = {}
        self._extensions: Dict[str, Any] = {}
        if extensions is not None:
            for ext in extensions:
                self.add_component(ext)

    def register_extension(self, name: str, extension: Any) -> None:
        """Register an extension with validation.
        通过验证注册扩展。

        Validates the extension and stores it in the extension registry.
        Also sets it as an attribute for backward compatibility.
        / 验证扩展并将其存储在扩展注册表中。
        还将其设置为属性以实现向后兼容性。

        Args:
            name: Unique identifier for the extension. / 扩展的唯一标识符。
            extension: Extension object to register. / 要注册的扩展对象。

        Raises:
            ValueError: If extension is None or name is empty.
                / 如果扩展为 None 或名称为空，则引发 ValueError。
        """
        if not name or name.strip() == "":
            raise ValueError("Extension name cannot be empty")
        if extension is None:
            raise ValueError("Extension cannot be None")
        self._extensions[name] = extension
        self.components[name] = extension
        setattr(self, name, extension)

    def get_extension(self, name: str) -> Any:
        """Get a registered extension by name.
        按名称获取已注册的扩展。

        Args:
            name: Identifier of the extension. / 扩展的标识符。

        Returns:
            The registered extension object. / 已注册的扩展对象。

        Raises:
            KeyError: If extension is not found.
                / 如果未找到扩展，则引发 KeyError。
        """
        if name not in self._extensions:
            raise KeyError(
                f"Extension '{name}' not found. "
                f"Available extensions: {list(self._extensions.keys())}"
            )
        return self._extensions[name]

    def has_extension(self, name: str) -> bool:
        """Check if an extension is registered.
        检查是否已注册扩展。

        Args:
            name: Identifier of the extension. / 扩展的标识符。

        Returns:
            True if extension is registered, False otherwise.
            / 如果已注册扩展则返回 True，否则返回 False。
        """
        return name in self._extensions

    def add_component(self, component: Any, descriptor: Optional[str] = None) -> None:
        """Add an extension component to the state.
        向状态添加扩展组件。

        This method is deprecated. Use register_extension() instead.
        / 此方法已弃用。改用 register_extension()。

        Args:
            component: Extension object to add. / 要添加的扩展对象。
            descriptor: Component identifier; defaults to component.descriptor.
                / 组件标识符；默认为 component.descriptor。
        """
        if descriptor is None:
            descriptor = component.descriptor
        self.register_extension(descriptor, component)

    def remove_component(self, descriptor: str) -> None:
        """Remove an extension component from the state.
        从状态移除扩展组件。

        Args:
            descriptor: Component identifier. / 组件标识符。
        """
        if descriptor in self._extensions:
            del self._extensions[descriptor]
        if descriptor in self.components:
            del self.components[descriptor]
        if hasattr(self, descriptor):
            delattr(self, descriptor)

    @property
    def kinetic_energy(self) -> float:
        """Compute total kinetic energy in eV.
        计算总动能（eV）。

        Returns:
            Kinetic energy: sum(0.5 * p_i^2 / m_i) in eV.
            / 动能：sum(0.5 * p_i^2 / m_i) 以 eV 为单位。
        """
        if self.p is None or self.m is None:
            return 0.0
        return 0.5 * cp.sum(self.p**2 / self.m[:, None]) * Constants.mv2_to_e

    @property
    def kinetic_virial(self) -> cp.ndarray:
        """Compute kinetic contribution to virial tensor in eV.
        计算动能对维里张量的贡献（eV）。

        Returns:
            3x3 kinetic virial tensor. / 3x3 动能维里张量。
        """
        if self.p is None or self.m is None:
            return cp.zeros((3, 3))
        return self.p.T @ (self.p / self.m[:, None]) * Constants.mv2_to_e

    @property
    def volume(self) -> float:
        """Compute simulation cell volume in Angstroms^3.
        计算模拟盒体积（立方埃）。

        Returns:
            Cell volume. / 盒体积。
        """
        if self.box is None:
            return 0.0
        return cp.linalg.det(self.box)

    def from_atoms(self, atoms: Atoms) -> "State":
        """Load state from an ASE Atoms object.
        从 ASE Atoms 对象加载状态。

        Args:
            atoms: ASE Atoms object. / ASE Atoms 对象。

        Returns:
            Self for method chaining. / 自身以便方法链接。
        """
        self.N = len(atoms)
        self.r = cp.array(atoms.get_positions())
        self.p = cp.array(atoms.get_momenta())
        self.m = cp.array(atoms.get_masses())
        self.box = cp.array(atoms.get_cell())
        self.atomic_number = cp.array(atoms.get_atomic_numbers())
        self.pbc = atoms.get_pbc()
        for ext in self.components.values():
            ext.from_atoms(atoms)
        return self

    def to_atoms(self) -> Atoms:
        """Convert state to an ASE Atoms object.
        将状态转换为 ASE Atoms 对象。

        Returns:
            ASE Atoms object. / ASE Atoms 对象。
        """
        atoms = Atoms(
            symbols=self.atomic_number,
            positions=cp.asnumpy(self.r),
            masses=cp.asnumpy(self.m),
            cell=cp.asnumpy(self.box),
            pbc=self.pbc,
        )
        for ext in self.components.values():
            ext.to_atoms(atoms)
        return atoms

    def configure_same_as(self, state2: Optional["State"]) -> bool:
        """Check if this state has the same configuration as another state.
        检查此状态是否与另一状态具有相同的配置。

        Verifies atom count, positions, cell parameters, and all components
        match within numerical tolerance.
        / 验证原子数、位置、盒参数和所有组件在数值容差范围内匹配。

        Args:
            state2: State to compare with. / 要比较的状态。

        Returns:
            True if configurations match. / 如果配置匹配则为真。
        """
        if state2 is None:
            return False
        if self.N != state2.N:
            return False
        if not cp.allclose(self.r, state2.r, atol=1e-7):
            return False
        if not cp.allclose(self.box, state2.box, atol=1e-7):
            return False
        for ext_name, ext in self.components.items():
            if ext_name not in state2.components:
                return False
            if not ext.configure_same_as(state2.components[ext_name]):
                return False
        return True

    def copy(self) -> "State":
        """Create a deep copy of the state.
        创建状态的深层副本。

        Returns:
            New State object with copied arrays. / 包含已复制数组的新 State 对象。
        """
        new_state = self.__class__(
            r=self.r.copy() if self.r is not None else None,
            p=self.p.copy() if self.p is not None else None,
            m=self.m.copy() if self.m is not None else None,
            box=self.box.copy() if self.box is not None else None,
            atomic_number=(
                self.atomic_number.copy() if self.atomic_number is not None else None
            ),
            pbc=self.pbc.copy() if self.pbc is not None else None,
        )
        for ext_name, ext in self.components.items():
            new_state.add_component(ext.copy(), descriptor=ext_name)
        return new_state


class Extension:
    """Base class for state extensions.
    状态扩展的基类。

    Extensions add auxiliary data to the simulation state, such as spin
    moments, thermostat variables, or barostat variables.
    / 扩展向模拟状态添加辅助数据，例如自旋矩、恒温器变量或恒压器变量。
    """

    descriptor: Optional[str] = None

    def from_atoms(self, atoms: Atoms) -> None:
        """Load extension data from ASE Atoms object.
        从 ASE Atoms 对象加载扩展数据。

        Args:
            atoms: ASE Atoms object. / ASE Atoms 对象。
        """
        ...

    def to_atoms(self, atoms: Atoms) -> None:
        """Store extension data in ASE Atoms object.
        在 ASE Atoms 对象中存储扩展数据。

        Args:
            atoms: ASE Atoms object to modify. / 要修改的 ASE Atoms 对象。
        """
        ...

    def copy(self) -> "Extension":
        """Create a copy of this extension.
        创建此扩展的副本。

        Returns:
            New Extension object. / 新扩展对象。
        """
        ...

    def configure_same_as(self, other: "Extension") -> bool:
        """Check if this extension has same configuration as another.
        检查此扩展是否与另一个扩展具有相同的配置。

        Args:
            other: Extension to compare with. / 要比较的扩展。

        Returns:
            True if configurations match. / 如果配置匹配则为真。
        """
        ...


class SpinExtension(Extension):
    """Extension for storing atomic spin moments.
    用于存储原子自旋矩的扩展。

    Stores spin moments as magnitude and direction vectors. Magnitude
    is stored separately to efficiently compute magnetic properties.
    / 将自旋矩存储为大小和方向向量。单独存储大小以有效计算磁性质。

    Attributes:
        magnitude: Spin magnitude for each atom, shape (N,).
            / 每个原子的自旋大小，形状 (N,)。
        vector: Spin direction (unit vectors), shape (N, 3).
            / 自旋方向（单位向量），形状 (N, 3)。
        inv_magnitude: Inverse magnitude for efficient calculations.
            / 用于有效计算的倒数大小。
    """

    descriptor: str = "spin"

    def __init__(self, spins: Optional[cp.ndarray] = None) -> None:
        """Initialize spin extension.
        初始化自旋扩展。

        Args:
            spins: Spin moment vectors in Bohr magnetons, shape (N, 3).
                / 自旋矩向量（Bohr 磁子），形状 (N, 3)。
        """
        self.magnitude: Optional[cp.ndarray] = None
        self.vector: Optional[cp.ndarray] = None
        self.inv_magnitude: Optional[cp.ndarray] = None
        self.spins = spins

    @property
    def spins(self) -> Optional[cp.ndarray]:
        """Get spin moment vectors.
        获取自旋矩向量。

        Returns:
            Spin moments in Bohr magnetons. / 自旋矩（Bohr 磁子）。
        """
        if self.vector is None or self.magnitude is None:
            return None
        return self.vector * self.magnitude[:, None]

    @spins.setter
    def spins(self, spins: Optional[cp.ndarray]) -> None:
        """Set spin moment vectors and decompose into magnitude and direction.
        设置自旋矩向量并分解为大小和方向。

        Args:
            spins: Spin moment vectors. / 自旋矩向量。
        """
        if spins is not None:
            self.magnitude = cp.linalg.norm(spins, axis=1)
            self.inv_magnitude = cp.where(
                self.magnitude > 0.01, 1.0 / self.magnitude, 0.0
            )
            self.vector = spins * self.inv_magnitude[:, None]

    @property
    def magnetic_moment(self) -> Optional[cp.ndarray]:
        """Compute total magnetic moment vector.
        计算总磁矩向量。

        Returns:
            Total magnetic moment in Bohr magnetons.
            / 总磁矩（Bohr 磁子）。
        """
        if self.spins is None:
            return None
        return cp.sum(self.spins, axis=0) / cp.sum(self.magnitude)

    def from_atoms(self, atoms: Atoms) -> None:
        """Load spins from ASE Atoms.info['spin'].
        从 ASE Atoms.info['spin'] 加载自旋。

        Args:
            atoms: ASE Atoms object. / ASE Atoms 对象。
        """
        self.spins = cp.array(atoms.info["spin"])

    def to_atoms(self, atoms: Atoms) -> None:
        """Store spins in ASE Atoms.info['spin'].
        在 ASE Atoms.info['spin'] 中存储自旋。

        Args:
            atoms: ASE Atoms object to modify. / 要修改的 ASE Atoms 对象。
        """
        atoms.info["spin"] = cp.asnumpy(self.spins)

    def configure_same_as(self, other: "SpinExtension") -> bool:
        """Check if spin configuration matches another extension.
        检查自旋配置是否与另一个扩展匹配。

        Args:
            other: Extension to compare with. / 要比较的扩展。

        Returns:
            True if spins are close. / 如果自旋接近则为真。
        """
        return cp.allclose(self.spins, other.spins, atol=1e-7)

    def copy(self) -> "SpinExtension":
        """Create a copy of this extension.
        创建此扩展的副本。

        Returns:
            New SpinExtension object. / 新 SpinExtension 对象。
        """
        new = self.__class__(spins=self.spins)
        return new

    def get_spin_temperature(
        self, state: State, context: Any, force_model: Any
    ) -> float:
        """Compute effective spin temperature from torques.
        从扭矩计算有效自旋温度。

        Uses the relationship between magnetic torque and thermal energy.
        Includes safety check for zero denominator.
        / 使用磁扭矩和热能之间的关系。
        包括零分母的安全检查。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context. / 模拟环境。
            force_model: Force model with spin_torques property.
                / 具有 spin_torques 属性的力模型。

        Returns:
            Effective spin temperature in Kelvin. / 有效自旋温度（开尔文）。
        """
        force_model.compute(state, context, properties=["spin_torques"])
        B_eff = force_model.results["spin_torques"]
        numerator = cp.sum(cp.linalg.norm(cp.cross(self.vector, B_eff), axis=1) ** 2)
        denominator = cp.sum(self.vector * B_eff)

        # Safe threshold to prevent division by zero
        # 安全阈值以防止除以零
        safe_threshold = 1e-12
        if cp.abs(denominator) < safe_threshold:
            return 0.0

        spin_temp = numerator / denominator / (2 * Constants.kB)
        return float(spin_temp)


class MTTKBarostatExtension(Extension):
    """Extension for MTTK (Martyna-Tobias-Tobias-Klein) barostat variables.
    MTTK（Martyna-Tobias-Tobias-Klein）恒压器变量的扩展。

    Stores variables needed for NPT ensemble dynamics including the
    cell strain rate (xi), cell momentum (p_xi), and related parameters.
    / 存储 NPT 系综动力学所需的变量，包括盒应变速率 (xi)、
    盒动量 (p_xi) 和相关参数。

    Attributes:
        W: Barostat mass parameter. / 恒压器质量参数。
        box_p: Box momentum conjugate to cell deformation.
            / 与盒变形共轭的盒动量。
        R: Current cell deformation matrix. / 当前盒变形矩阵。
        xi: Cell strain rate. / 盒应变速率。
        p_xi: Momentum conjugate to xi. / 与 xi 共轭的动量。
    """

    descriptor: str = "mttk_barostat"

    def __init__(
        self,
        W: Optional[cp.ndarray] = None,
        box_p: Optional[cp.ndarray] = None,
        R: Optional[cp.ndarray] = None,
        xi: Optional[cp.ndarray] = None,
        p_xi: Optional[cp.ndarray] = None,
    ) -> None:
        """Initialize MTTK barostat extension.
        初始化 MTTK 恒压器扩展。

        Args:
            W: Barostat mass. / 恒压器质量。
            box_p: Box momentum. / 盒动量。
            R: Cell deformation matrix. / 盒变形矩阵。
            xi: Cell strain rate. / 盒应变速率。
            p_xi: Momentum conjugate to xi. / 与 xi 共轭的动量。
        """
        self.W: Optional[cp.ndarray] = W
        self.box_p: Optional[cp.ndarray] = box_p
        self.R: Optional[cp.ndarray] = R
        self.xi: Optional[cp.ndarray] = xi
        self.p_xi: Optional[cp.ndarray] = p_xi

    def copy(self) -> "MTTKBarostatExtension":
        """Create a copy of this extension.
        创建此扩展的副本。

        Returns:
            New MTTKBarostatExtension object. / 新 MTTKBarostatExtension 对象。
        """
        new = self.__class__(
            W=self.W.copy() if self.W is not None else None,
            box_p=self.box_p.copy() if self.box_p is not None else None,
            R=self.R.copy() if self.R is not None else None,
            xi=self.xi.copy() if self.xi is not None else None,
            p_xi=self.p_xi.copy() if self.p_xi is not None else None,
        )
        return new

    def configure_same_as(self, other: "MTTKBarostatExtension") -> bool:
        """Check if barostat configuration matches another extension.
        检查恒压器配置是否与另一个扩展匹配。

        For barostats, exact matching is not required as they equilibrate.
        / 对于恒压器，不需要精确匹配，因为它们会平衡。

        Args:
            other: Extension to compare with. / 要比较的扩展。

        Returns:
            Always True for barostat compatibility. / 对于恒压器兼容性始终为真。
        """
        return True


class NHThermostatExtension(Extension):
    """Extension for Nose-Hoover thermostat variables.
    Nose-Hoover 恒温器变量的扩展。

    Stores variables for the Nose-Hoover thermostat chain that maintains
    constant temperature in NVT simulations.
    / 存储 Nose-Hoover 恒温器链的变量，用于在 NVT 模拟中保持恒定温度。

    Attributes:
        Q: Thermostat mass parameters for chain, shape (M,).
            / 链的恒温器质量参数，形状 (M,)。
        eta: Thermostat position variables for chain, shape (M,).
            / 链的恒温器位置变量，形状 (M,)。
        p_eta: Thermostat momentum variables for chain, shape (M,).
            / 链的恒温器动量变量，形状 (M,)。
    """

    descriptor: str = "nh_thermostat"

    def __init__(
        self,
        Q: Optional[cp.ndarray] = None,
        eta: Optional[cp.ndarray] = None,
        p_eta: Optional[cp.ndarray] = None,
    ) -> None:
        """Initialize Nose-Hoover thermostat extension.
        初始化 Nose-Hoover 恒温器扩展。

        Args:
            Q: Thermostat mass parameters. / 恒温器质量参数。
            eta: Thermostat position variables. / 恒温器位置变量。
            p_eta: Thermostat momentum variables. / 恒温器动量变量。
        """
        self.Q: Optional[cp.ndarray] = Q
        self.eta: Optional[cp.ndarray] = eta
        self.p_eta: Optional[cp.ndarray] = p_eta

    def copy(self) -> "NHThermostatExtension":
        """Create a copy of this extension.
        创建此扩展的副本。

        Returns:
            New NHThermostatExtension object. / 新 NHThermostatExtension 对象。
        """
        new = self.__class__(
            Q=self.Q.copy() if self.Q is not None else None,
            eta=self.eta.copy() if self.eta is not None else None,
            p_eta=self.p_eta.copy() if self.p_eta is not None else None,
        )
        return new

    def configure_same_as(self, other: "NHThermostatExtension") -> bool:
        """Check if thermostat configuration matches another extension.
        检查恒温器配置是否与另一个扩展匹配。

        For thermostats, exact matching is not required as they equilibrate.
        / 对于恒温器，不需要精确匹配，因为它们会平衡。

        Args:
            other: Extension to compare with. / 要比较的扩展。

        Returns:
            Always True for thermostat compatibility. / 对于恒温器兼容性始终为真。
        """
        return True

    def get_thermostat_energy(self, state: State, context: Any) -> float:
        """Compute thermostat chain energy contribution.
        计算恒温器链能量贡献。

        Returns the energy-like contribution from thermostat variables
        according to Martyna, Tuckerman, Tobias, Klein (Mol Phys, 87, 1117).
        Thermostat chain energy = Sum(0.5*p_eta_k^2/Q_k) + L*k*T*eta_1
        + Sum(k*T*eta_k), where L = tdof (total degrees of freedom).
        / 根据 Martyna、Tuckerman、Tobias、Klein（Mol Phys、87、1117）
        返回恒温器变量的类能量贡献。恒温器链能量 = Sum(0.5*p_eta_k^2/Q_k)
        + L*k*T*eta_1 + Sum(k*T*eta_k)，其中 L = tdof（总自由度）。

        Args:
            state: Simulation state. / 模拟状态。
            context: Simulation context with target_temp.
                / 带有 target_temp 的模拟环境。

        Returns:
            Thermostat energy contribution in eV.
            / 恒温器能量贡献（eV）。
        """
        kT = Constants.kB * context.target_temp
        energy = (
            3 * state.N * kT * self.eta[0]
            + kT * cp.sum(self.eta[1:])
            + cp.sum(0.5 * self.p_eta**2 / self.Q)
        )
        return float(energy)
