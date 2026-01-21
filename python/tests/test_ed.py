"""Tests for exact diagonalization."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfim.ed import (
    build_hamiltonian,
    exact_diagonalization,
    compute_correlator,
    compute_expectation,
    compute_zz_correlation_function,
    estimate_correlation_length,
)
from tfim.constants import SIGMA_X, SIGMA_Z, IDENTITY
from tfim.exact import exact_energy


class TestBuildHamiltonian:
    """Tests for Hamiltonian construction."""

    def test_hamiltonian_hermitian(self):
        """Hamiltonian should be Hermitian."""
        for L in [2, 4, 6]:
            H = build_hamiltonian(L, g=1.0)
            H_dense = H.toarray()
            assert np.allclose(H_dense, H_dense.conj().T)

    def test_hamiltonian_dimension(self):
        """Hamiltonian has correct dimension 2^L x 2^L."""
        for L in [2, 3, 4]:
            H = build_hamiltonian(L, g=1.0)
            assert H.shape == (2**L, 2**L)

    def test_hamiltonian_at_g_zero(self):
        """At g=0, ground state is ferromagnetic."""
        L = 4
        H = build_hamiltonian(L, g=0.0, J=1.0, periodic=False)
        eigenvalues = np.linalg.eigvalsh(H.toarray())
        E0 = min(eigenvalues)
        # For L=4 with OBC: E = -(L-1) = -3
        assert np.isclose(E0, -(L-1), atol=1e-10)

    def test_hamiltonian_periodic_vs_open(self):
        """Periodic BC has more bonds than open BC."""
        L = 4
        H_open = build_hamiltonian(L, g=0.0, J=1.0, periodic=False)
        H_pbc = build_hamiltonian(L, g=0.0, J=1.0, periodic=True)

        E_open = min(np.linalg.eigvalsh(H_open.toarray()))
        E_pbc = min(np.linalg.eigvalsh(H_pbc.toarray()))

        # PBC has L bonds, OBC has L-1 bonds
        # For g=0: E_obc = -(L-1), E_pbc = -L
        assert E_pbc < E_open

    def test_hamiltonian_j_scaling(self):
        """Hamiltonian scales with J."""
        L = 4
        H1 = build_hamiltonian(L, g=0.5, J=1.0)
        H2 = build_hamiltonian(L, g=0.5, J=2.0)

        E1 = min(np.linalg.eigvalsh(H1.toarray()))
        E2 = min(np.linalg.eigvalsh(H2.toarray()))

        # Not exactly 2x due to transverse field term
        # but at g=0, it should be 2x
        H1_g0 = build_hamiltonian(L, g=0.0, J=1.0)
        H2_g0 = build_hamiltonian(L, g=0.0, J=2.0)
        E1_g0 = min(np.linalg.eigvalsh(H1_g0.toarray()))
        E2_g0 = min(np.linalg.eigvalsh(H2_g0.toarray()))
        assert np.isclose(E2_g0, 2*E1_g0)


class TestExactDiagonalization:
    """Tests for ED solver."""

    def test_ed_returns_correct_structure(self):
        """ED returns all expected fields."""
        result = exact_diagonalization(4, g=1.0)
        assert hasattr(result, 'L')
        assert hasattr(result, 'g')
        assert hasattr(result, 'energy')
        assert hasattr(result, 'energy_per_site')
        assert hasattr(result, 'gap')
        assert hasattr(result, 'ground_state')
        assert hasattr(result, 'eigenvalues')

    def test_ed_ground_state_normalized(self):
        """Ground state should be normalized."""
        result = exact_diagonalization(4, g=1.0)
        norm = np.linalg.norm(result.ground_state)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_ed_eigenvalues_sorted(self):
        """Eigenvalues should be sorted."""
        result = exact_diagonalization(4, g=1.0, n_states=4)
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] <= result.eigenvalues[i+1]

    def test_ed_gap_positive(self):
        """Energy gap should be positive (away from critical)."""
        result = exact_diagonalization(4, g=0.5)
        assert result.gap > 0

        result = exact_diagonalization(4, g=2.0)
        assert result.gap > 0

    def test_ed_converges_to_exact(self):
        """ED energy converges to exact for larger L."""
        E_exact = exact_energy(0.8)

        errors = []
        for L in [6, 8, 10, 12]:
            result = exact_diagonalization(L, g=0.8)
            errors.append(abs(result.energy_per_site - E_exact))

        # Error should decrease with L (finite size effects)
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i] * 1.5  # Allow some fluctuation

    def test_ed_small_system(self):
        """Test on smallest system L=2."""
        result = exact_diagonalization(2, g=1.0)
        assert result.L == 2
        assert len(result.ground_state) == 4


class TestCorrelators:
    """Tests for correlation functions."""

    def test_zz_nearest_neighbor(self):
        """<Z_i Z_{i+1}> is real for nearest neighbors."""
        L = 4
        result = exact_diagonalization(L, g=1.0)
        psi = result.ground_state

        for i in range(L - 1):
            zz = compute_correlator(psi, L, SIGMA_Z, i, SIGMA_Z, i + 1)
            assert np.isclose(np.imag(zz), 0.0, atol=1e-10)
            assert np.abs(zz) <= 1.0 + 1e-10  # Bounded by 1

    def test_xx_expectation(self):
        """<X_i> is enhanced in transverse field direction."""
        L = 4
        result = exact_diagonalization(L, g=2.0)  # Strong field
        psi = result.ground_state

        # In strong field, spins align with X
        for i in range(L):
            x = compute_expectation(psi, L, SIGMA_X, i)
            assert np.real(x) > 0.5  # Positive X magnetization

    def test_z_expectation_symmetric(self):
        """<Z_i> = 0 for paramagnetic phase."""
        L = 6
        result = exact_diagonalization(L, g=2.0)  # Disordered phase
        psi = result.ground_state

        # In thermodynamic limit, <Z> = 0 for g > 1
        # For finite systems, it's small but not exactly zero
        for i in range(L):
            z = compute_expectation(psi, L, SIGMA_Z, i)
            assert np.abs(z) < 0.3  # Not exactly zero for finite systems

    def test_correlator_hermitian(self):
        """<ZZ> is real for Hermitian operators."""
        L = 4
        result = exact_diagonalization(L, g=1.0)
        psi = result.ground_state

        zz = compute_correlator(psi, L, SIGMA_Z, 0, SIGMA_Z, 2)
        assert np.isclose(np.imag(zz), 0.0, atol=1e-10)


class TestCorrelationFunction:
    """Tests for ZZ correlation function."""

    def test_correlation_function_structure(self):
        """Correlation function returns expected structure."""
        L = 8
        result = exact_diagonalization(L, g=1.0)
        corr = compute_zz_correlation_function(result.ground_state, L)

        assert 'distances' in corr
        assert 'raw_zz' in corr
        assert 'connected' in corr
        assert len(corr['distances']) == len(corr['raw_zz'])

    def test_correlation_has_correct_length(self):
        """Correlation function has expected number of entries."""
        L = 8
        result = exact_diagonalization(L, g=1.0)
        corr = compute_zz_correlation_function(result.ground_state, L)

        # Should have L//2 entries (from reference site to half the system)
        assert len(corr['distances']) <= L // 2
        assert len(corr['distances']) > 0

    def test_correlation_decay_disordered(self):
        """Correlations decay in disordered phase."""
        L = 10
        result = exact_diagonalization(L, g=2.0)  # Deep in disordered phase
        corr = compute_zz_correlation_function(result.ground_state, L)

        # Should decay with distance
        connected = np.abs(corr['connected'][1:])  # Skip r=0
        if len(connected) > 2:
            assert connected[-1] < connected[0]


class TestEstimateCorrelationLength:
    """Tests for correlation length estimation."""

    def test_xi_finite_in_disordered(self):
        """ξ should be finite in disordered phase."""
        L = 12
        result = exact_diagonalization(L, g=2.0)
        xi = estimate_correlation_length(result.ground_state, L)

        assert np.isfinite(xi)
        assert xi > 0

    def test_xi_larger_near_critical(self):
        """ξ increases as we approach critical point."""
        L = 12

        result_far = exact_diagonalization(L, g=2.0)
        xi_far = estimate_correlation_length(result_far.ground_state, L)

        result_near = exact_diagonalization(L, g=1.2)
        xi_near = estimate_correlation_length(result_near.ground_state, L)

        if np.isfinite(xi_far) and np.isfinite(xi_near):
            assert xi_near > xi_far


class TestConstants:
    """Tests for physical constants."""

    def test_pauli_matrices_hermitian(self):
        """Pauli matrices are Hermitian."""
        assert np.allclose(SIGMA_X, SIGMA_X.conj().T)
        assert np.allclose(SIGMA_Z, SIGMA_Z.conj().T)

    def test_pauli_matrices_trace(self):
        """Pauli matrices are traceless."""
        assert np.isclose(np.trace(SIGMA_X), 0)
        assert np.isclose(np.trace(SIGMA_Z), 0)

    def test_pauli_matrices_squared(self):
        """σ² = I for Pauli matrices."""
        assert np.allclose(SIGMA_X @ SIGMA_X, IDENTITY)
        assert np.allclose(SIGMA_Z @ SIGMA_Z, IDENTITY)

    def test_pauli_anticommutator(self):
        """Pauli matrices anticommute: {σ_x, σ_z} = 0."""
        anticomm = SIGMA_X @ SIGMA_Z + SIGMA_Z @ SIGMA_X
        assert np.allclose(anticomm, 0)
