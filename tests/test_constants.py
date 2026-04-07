"""Physical constants verification tests.
物理常数验证测试。

Tests verify that the physical constants used in the simulation match
expected values from CODATA 2018 and related standards, and that unit
conversion factors are mathematically consistent.
"""

import pytest
import math
from pydyn.constants import Constants


class TestBoltzmannConstant:
    """Test Boltzmann constant value.
    测试 Boltzmann 常数值。
    """

    def test_kb_value(self):
        """Verify kB matches CODATA 2018 value (8.617333262145e-5 eV/K).
        验证 kB 匹配 CODATA 2018 值 (8.617333262145e-5 eV/K)。
        """
        expected_kb = 8.617333262145e-5  # eV/K from CODATA 2018
        # 来自 CODATA 2018 的值 eV/K
        assert Constants.kB == expected_kb, \
            f"Expected {expected_kb}, got {Constants.kB}"

    def test_kb_positive(self):
        """Verify kB is positive.
        验证 kB 为正。
        """
        assert Constants.kB > 0, "Boltzmann constant must be positive"


class TestPlanckConstant:
    """Test Planck constant values.
    测试 Planck 常数值。
    """

    def test_planck_positive(self):
        """Verify Planck constant is positive.
        验证 Planck 常数为正。
        """
        assert Constants.hplanck > 0, "Planck constant must be positive"

    def test_hbar_consistency(self):
        """Verify hbar = h / (2*pi).
        验证 hbar = h / (2*pi)。
        """
        expected_hbar = Constants.hplanck / (2 * Constants.pi)
        assert abs(Constants.hbar - expected_hbar) < 1e-15, \
            f"hbar inconsistency: {Constants.hbar} vs {expected_hbar}"


class TestEnergyConversionFactors:
    """Test energy unit conversion factors.
    测试能量单位转换因子。
    """

    def test_mv2_to_e_positive(self):
        """Verify mv2_to_e is positive.
        验证 mv2_to_e 为正。
        """
        assert Constants.mv2_to_e > 0, \
            "Conversion factor mv2_to_e must be positive"

    def test_e_to_mv2_positive(self):
        """Verify e_to_mv2 is positive.
        验证 e_to_mv2 为正。
        """
        assert Constants.e_to_mv2 > 0, \
            "Conversion factor e_to_mv2 must be positive"

    def test_mv2_e_are_inverses(self):
        """Verify mv2_to_e and e_to_mv2 are inverses.
        验证 mv2_to_e 和 e_to_mv2 是倒数。

        The product should equal 1.0 (within numerical precision).
        / 乘积应等于 1.0（在数值精度范围内）。
        """
        product = Constants.mv2_to_e * Constants.e_to_mv2
        assert abs(product - 1.0) < 1e-10, \
            f"Conversion factors not inverses: {product} != 1.0"

    def test_mv2_to_e_value(self):
        """Verify mv2_to_e has expected value.
        验证 mv2_to_e 具有预期值。

        The expected value is 1.0364269e-4 for amu*Ang^2/ps^2 to eV.
        / 对于 amu*Ang^2/ps^2 到 eV，预期值为 1.0364269e-4。
        """
        expected = 1.0364269e-4
        assert abs(Constants.mv2_to_e - expected) < 1e-12, \
            f"Expected {expected}, got {Constants.mv2_to_e}"


class TestPressureVolumeConversion:
    """Test pressure-volume unit conversion factors.
    测试压力-体积单位转换因子。
    """

    def test_e_to_pV_positive(self):
        """Verify e_to_pV is positive.
        验证 e_to_pV 为正。
        """
        assert Constants.e_to_pV > 0, \
            "Conversion factor e_to_pV must be positive"

    def test_pV_to_e_positive(self):
        """Verify pV_to_e is positive.
        验证 pV_to_e 为正。
        """
        assert Constants.pV_to_e > 0, \
            "Conversion factor pV_to_e must be positive"

    def test_pV_e_are_inverses(self):
        """Verify e_to_pV and pV_to_e are inverses.
        验证 e_to_pV 和 pV_to_e 是倒数。

        The product should equal 1.0 (within numerical precision).
        / 乘积应等于 1.0（在数值精度范围内）。
        """
        product = Constants.e_to_pV * Constants.pV_to_e
        assert abs(product - 1.0) < 1e-10, \
            f"Conversion factors not inverses: {product} != 1.0"

    def test_e_to_pV_value(self):
        """Verify e_to_pV has expected value.
        验证 e_to_pV 具有预期值。

        The expected value is 1.6021766208e6 for eV to bar*Ang^3.
        / 对于 eV 到 bar*Ang^3，预期值为 1.6021766208e6。
        """
        expected = 1.6021766208e6
        assert abs(Constants.e_to_pV - expected) < 1e-2, \
            f"Expected {expected}, got {Constants.e_to_pV}"


class TestMagneticConstants:
    """Test magnetic constants.
    测试磁常数。
    """

    def test_bohr_magneton_positive(self):
        """Verify Bohr magneton is positive.
        验证 Bohr 磁子为正。
        """
        assert Constants.mu_B > 0, "Bohr magneton must be positive"

    def test_gyromagnetic_ratio_positive(self):
        """Verify gyromagnetic ratio is positive.
        验证旋磁比为正。
        """
        assert Constants.gamma > 0, "Gyromagnetic ratio must be positive"

    def test_magnetic_constants_magnitude(self):
        """Verify magnetic constants have reasonable values.
        验证磁常数具有合理的值。
        """
        # mu_B should be ~5.79e-5 eV/T
        # mu_B 应该是 ~5.79e-5 eV/T
        assert 1e-5 < Constants.mu_B < 1e-4, \
            f"Bohr magneton magnitude unexpected: {Constants.mu_B}"

        # gamma should be ~0.176 rad/(ps*T)
        # gamma 应该是 ~0.176 rad/(ps*T)
        assert 0.1 < Constants.gamma < 0.2, \
            f"Gyromagnetic ratio magnitude unexpected: {Constants.gamma}"


class TestAtomicMassUnit:
    """Test atomic mass unit constant.
    测试原子质量单位常数。
    """

    def test_amu_positive(self):
        """Verify AMU is positive.
        验证 AMU 为正。
        """
        assert Constants.amu > 0, "Atomic mass unit must be positive"

    def test_amu_value(self):
        """Verify amu has expected value in kg.
        验证 amu 在 kg 中具有预期值。

        The expected value is 1.6605390689e-27 kg (CODATA 2018).
        / 预期值为 1.6605390689e-27 kg (CODATA 2018)。
        """
        expected = 1.6605390689e-27
        assert abs(Constants.amu - expected) < 1e-36, \
            f"Expected {expected}, got {Constants.amu}"


class TestMathematicalConstant:
    """Test mathematical constants.
    测试数学常数。
    """

    def test_pi_value(self):
        """Verify pi has correct value.
        验证 pi 具有正确值。
        """
        expected = math.pi
        assert abs(Constants.pi - expected) < 1e-15, \
            f"Pi constant incorrect: {Constants.pi} vs {expected}"

    def test_pi_positive(self):
        """Verify pi is positive.
        验证 pi 为正。
        """
        assert Constants.pi > 0, "Pi must be positive"


class TestConversionChain:
    """Test that conversion factors compose correctly.
    测试转换因子能否正确组合。
    """

    def test_energy_roundtrip_mv2(self):
        """Test roundtrip conversion: mv2 -> eV -> mv2.
        测试往返转换：mv2 -> eV -> mv2。

        A value converted forward and back should equal the original.
        / 向前和向后转换的值应该等于原始值。
        """
        original = 10.0  # Some arbitrary energy in amu*Ang^2/ps^2
        # 以 amu*Ang^2/ps^2 为单位的某个任意能量
        converted_to_ev = original * Constants.mv2_to_e
        back_to_mv2 = converted_to_ev * Constants.e_to_mv2
        assert abs(back_to_mv2 - original) < 1e-12, \
            f"Roundtrip conversion failed: {original} -> {back_to_mv2}"

    def test_energy_roundtrip_pv(self):
        """Test roundtrip conversion: pV -> eV -> pV.
        测试往返转换：pV -> eV -> pV。

        A value converted forward and back should equal the original.
        / 向前和向后转换的值应该等于原始值。
        """
        original = 1e6  # Some arbitrary pressure-volume product
        # 某个任意的压力-体积乘积
        converted_to_ev = original * Constants.pV_to_e
        back_to_pv = converted_to_ev * Constants.e_to_pV
        assert abs(back_to_pv - original) < 1.0, \
            f"Roundtrip conversion failed: {original} -> {back_to_pv}"
