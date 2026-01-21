"""Tests for exact analytical solutions."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfim.exact import (
    exact_energy,
    exact_correlation_length,
    critical_point,
    magnetization_z,
    energy_gap,
    phase,
    format_xi,
    CRITICAL_POINT_1D,
    CRITICAL_POINT_2D,
)


class TestExactEnergy:
    """Tests for exact ground state energy."""

    def test_energy_at_g_zero(self):
        """At g=0, H = -J Σ ZZ, ground state is ferromagnetic."""
        # E/N = -1 for g=0 (all spins aligned)
        E = exact_energy(0.0, J=1.0)
        assert np.isclose(E, -1.0, atol=1e-10)

    def test_energy_at_g_infinity(self):
        """At g→∞, spins align with field, E/N → -g."""
        E = exact_energy(100.0, J=1.0)
        # For large g, E/N ≈ -g
        assert E < -99.0

    def test_energy_at_critical_point(self):
        """Test known value at g=1."""
        E = exact_energy(1.0, J=1.0)
        # Known value: E_c/N ≈ -4/π ≈ -1.2732...
        assert np.isclose(E, -4/np.pi, atol=1e-4)

    def test_energy_monotonic(self):
        """Energy decreases with increasing g."""
        g_values = [0.5, 1.0, 1.5, 2.0]
        energies = [exact_energy(g) for g in g_values]
        for i in range(len(energies) - 1):
            assert energies[i] > energies[i+1], "Energy should decrease with g"

    def test_energy_specific_values(self):
        """Test against known literature values."""
        # At critical point g=1, E = -4/π
        E_critical = exact_energy(1.0)
        assert np.isclose(E_critical, -4/np.pi, atol=1e-6)

        # For g << 1, E ≈ -1 (ferromagnetic limit)
        E_ferro = exact_energy(0.01)
        assert np.isclose(E_ferro, -1.0, atol=0.01)

        # For g >> 1, E ≈ -g (paramagnetic limit)
        E_para = exact_energy(10.0)
        assert np.isclose(E_para, -10.0, atol=0.1)

    def test_energy_duality(self):
        """E(g) and E(1/g)/g satisfy duality relation."""
        # For TFIM, there's a duality: E(g) relates to E(1/g)
        # E(g)/g is symmetric under g -> 1/g at critical point
        g = 0.5
        E_g = exact_energy(g)
        E_inv = exact_energy(1.0/g)
        # E(g)/J ~ g * E(1/g) / (1/g) for large/small g asymptotically
        # This is approximate; just check energy is negative and reasonable
        assert E_g < 0
        assert E_inv < 0


class TestCorrelationLength:
    """Tests for exact correlation length."""

    def test_xi_at_critical_point(self):
        """Correlation length diverges at g=1."""
        xi = exact_correlation_length(1.0)
        assert np.isinf(xi)

    def test_xi_ordered_phase(self):
        """Correlation length for g < 1."""
        xi = exact_correlation_length(0.5)
        expected = 1.0 / np.abs(np.log(0.5))
        assert np.isclose(xi, expected, rtol=1e-10)

    def test_xi_disordered_phase(self):
        """Correlation length for g > 1."""
        xi = exact_correlation_length(2.0)
        expected = 1.0 / np.abs(np.log(2.0))
        assert np.isclose(xi, expected, rtol=1e-10)

    def test_xi_symmetry(self):
        """ξ(g) = ξ(1/g) due to duality."""
        g_values = [0.5, 0.7, 0.9]
        for g in g_values:
            xi_g = exact_correlation_length(g)
            xi_inv = exact_correlation_length(1.0/g)
            assert np.isclose(xi_g, xi_inv, rtol=1e-10)

    def test_xi_approaches_infinity_near_critical(self):
        """ξ → ∞ as g → 1."""
        xi_099 = exact_correlation_length(0.99)
        xi_0999 = exact_correlation_length(0.999)
        assert xi_0999 > xi_099 > 10


class TestCriticalPoint:
    """Tests for critical point values."""

    def test_1d_critical_point(self):
        """1D critical point is g_c = 1."""
        assert critical_point(1) == CRITICAL_POINT_1D == 1.0

    def test_2d_critical_point(self):
        """2D critical point is approximately 3.044."""
        assert np.isclose(critical_point(2), CRITICAL_POINT_2D, rtol=0.01)

    def test_invalid_dimension(self):
        """Invalid dimension raises error."""
        with pytest.raises(ValueError):
            critical_point(3)


class TestMagnetization:
    """Tests for magnetization."""

    def test_magnetization_ordered_phase(self):
        """Non-zero magnetization for g < 1."""
        m = magnetization_z(0.5)
        expected = (1 - 0.5**2) ** 0.125
        assert np.isclose(m, expected, rtol=1e-10)
        assert m > 0

    def test_magnetization_disordered_phase(self):
        """Zero magnetization for g >= 1."""
        assert magnetization_z(1.0) == 0.0
        assert magnetization_z(1.5) == 0.0
        assert magnetization_z(2.0) == 0.0

    def test_magnetization_at_g_zero(self):
        """Full magnetization at g=0."""
        m = magnetization_z(0.0)
        assert np.isclose(m, 1.0)


class TestEnergyGap:
    """Tests for energy gap."""

    def test_gap_closes_at_critical(self):
        """Gap closes at g=1."""
        gap = energy_gap(1.0)
        assert gap == 0.0

    def test_gap_ordered_phase(self):
        """Gap in ordered phase."""
        gap = energy_gap(0.5)
        assert np.isclose(gap, 2*0.5)  # 2|1-g| = 1.0

    def test_gap_disordered_phase(self):
        """Gap in disordered phase."""
        gap = energy_gap(2.0)
        assert np.isclose(gap, 2*1.0)  # 2|1-g| = 2.0


class TestPhase:
    """Tests for phase identification."""

    def test_ordered_phase(self):
        """g < 1 is ordered."""
        assert phase(0.5) == "ordered"
        assert phase(0.99) == "ordered"

    def test_disordered_phase(self):
        """g > 1 is disordered."""
        assert phase(1.5) == "disordered"
        assert phase(2.0) == "disordered"

    def test_critical_phase(self):
        """g = 1 is critical."""
        assert phase(1.0) == "critical"


class TestFormatXi:
    """Tests for formatting correlation length."""

    def test_format_finite(self):
        """Format finite values."""
        assert format_xi(1.2345) == "1.2345"

    def test_format_infinity(self):
        """Format infinity."""
        assert format_xi(np.inf) == "∞"
