"""Physical and mathematical constants for molecular dynamics simulations.
分子动力学模拟的物理和数学常数。
"""


class Constants:
    """Collection of physical constants used in molecular dynamics.
    分子动力学中使用的物理常数集合。

    Provides SI and atomic unit constants needed for unit conversions,
    thermal calculations, and magnetic properties. All energy units are eV.
    / 提供单位转换、热计算和磁性质所需的 SI 和原子单位常数。
    所有能量单位均为 eV。
    """

    # Mathematical constant / 数学常数
    pi: float = 3.141592653589793
    """The value of Pi. / Pi 的值。"""

    # Magnetic constants / 磁常数
    mu_B: float = 5.7883817555e-5
    """Bohr Magneton in eV/T. / Bohr 磁子（eV/T）。"""

    mu_0: float = 2.0133545 * 1e-25
    """Vacuum permeability in T^2*m^3/eV. / 真空磁导率（T^2*m^3/eV）。"""

    gamma: float = 0.1760859644
    """Gyromagnetic ratio of electron in rad/(ps*T).
    / 电子的旋磁比（rad/(ps*T)）。"""

    # Thermal constants / 热常数
    kB: float = 8.617333262145e-5
    """Boltzmann constant in eV/K. / Boltzmann 常数（eV/K）。"""

    hplanck: float = 4.13566733e-3
    """Planck constant in eV*ps. / Planck 常数（eV*ps）。"""

    # Derived constants / 派生常数
    hbar: float = hplanck / (2 * pi)
    """Reduced Planck constant (hbar) in eV*ps.
    / 约化 Planck 常数（hbar）（eV*ps）。"""

    amu: float = 1.6605390689e-27
    """Atomic mass unit in kg. / 原子质量单位（kg）。"""

    # Unit conversion constants / 单位转换常数
    mv2_to_e: float = 1.0364269e-4
    """Conversion factor: amu*Angstrom^2/ps^2 to eV.
    / 转换因子：amu*埃^2/ps^2 到 eV。"""

    e_to_mv2: float = 1 / 1.0364269e-4
    """Conversion factor: eV to amu*Angstrom^2/ps^2.
    / 转换因子：eV 到 amu*埃^2/ps^2。"""

    e_to_pV: float = 1.6021766208e6
    """Conversion factor: eV to bar*Angstrom^3.
    / 转换因子：eV 到 bar*埃^3。"""

    pV_to_e: float = 1 / 1.6021766208e6
    """Conversion factor: bar*Angstrom^3 to eV.
    / 转换因子：bar*埃^3 到 eV。"""
