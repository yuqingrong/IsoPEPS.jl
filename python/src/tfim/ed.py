"""
Exact diagonalization for finite TFIM systems.

Uses scipy sparse eigensolvers for systems up to L ~ 18-20 sites.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .constants import SIGMA_X, SIGMA_Z, IDENTITY


@dataclass
class EDResult:
    """Result from exact diagonalization."""
    L: int
    g: float
    J: float
    periodic: bool
    energy: float
    energy_per_site: float
    gap: float
    eigenvalues: np.ndarray
    ground_state: np.ndarray


def _single_site_operator(op: np.ndarray, site: int, L: int) -> sparse.csr_matrix:
    """Build operator acting on a single site in L-site system."""
    ops = [sparse.csr_matrix(IDENTITY) for _ in range(L)]
    ops[site] = sparse.csr_matrix(op)
    result = ops[0]
    for k in range(1, L):
        result = sparse.kron(result, ops[k], format='csr')
    return result


def _two_site_operator(op1: np.ndarray, site1: int,
                       op2: np.ndarray, site2: int, L: int) -> sparse.csr_matrix:
    """Build operator acting on two sites in L-site system."""
    ops = [sparse.csr_matrix(IDENTITY) for _ in range(L)]
    ops[site1] = sparse.csr_matrix(op1)
    ops[site2] = sparse.csr_matrix(op2)
    result = ops[0]
    for k in range(1, L):
        result = sparse.kron(result, ops[k], format='csr')
    return result


def build_hamiltonian(L: int, g: float, J: float = 1.0,
                      periodic: bool = False) -> sparse.csr_matrix:
    """
    Build sparse TFIM Hamiltonian.

    H = -J Σ Z_i Z_{i+1} - g Σ X_i

    Args:
        L: Number of sites
        g: Transverse field strength (h/J)
        J: Coupling strength
        periodic: Use periodic boundary conditions

    Returns:
        Sparse Hamiltonian matrix (2^L x 2^L)
    """
    dim = 2**L
    H = sparse.csr_matrix((dim, dim), dtype=complex)

    # Transverse field terms: -g Σ X_i
    for i in range(L):
        H -= g * _single_site_operator(SIGMA_X, i, L)

    # ZZ coupling terms: -J Σ Z_i Z_{i+1}
    n_bonds = L if periodic else L - 1
    for i in range(n_bonds):
        j = (i + 1) % L
        H -= J * _two_site_operator(SIGMA_Z, i, SIGMA_Z, j, L)

    return H


def exact_diagonalization(L: int, g: float, J: float = 1.0,
                          periodic: bool = False,
                          n_states: int = 4) -> EDResult:
    """
    Compute ground state via exact diagonalization.

    Args:
        L: Number of sites
        g: Transverse field strength
        J: Coupling strength
        periodic: Periodic boundary conditions
        n_states: Number of low-lying states to compute

    Returns:
        EDResult with energy, gap, and ground state vector
    """
    H = build_hamiltonian(L, g, J, periodic)

    # Find lowest eigenvalues
    k = min(n_states, 2**L - 2)
    eigenvalues, eigenvectors = eigsh(H, k=k, which='SA', return_eigenvectors=True)

    # Sort by energy
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    E0 = eigenvalues[0]
    E1 = eigenvalues[1] if len(eigenvalues) > 1 else np.nan
    gap = E1 - E0

    return EDResult(
        L=L,
        g=g,
        J=J,
        periodic=periodic,
        energy=E0,
        energy_per_site=E0 / L,
        gap=gap,
        eigenvalues=eigenvalues,
        ground_state=eigenvectors[:, 0],
    )


def compute_correlator(psi: np.ndarray, L: int, op1: np.ndarray,
                       site1: int, op2: np.ndarray, site2: int) -> complex:
    """
    Compute two-point correlator <psi| O_1 O_2 |psi>.

    Args:
        psi: State vector (2^L)
        L: Number of sites
        op1, op2: Single-site operators (2x2 matrices)
        site1, site2: Site indices

    Returns:
        Complex expectation value
    """
    O = _two_site_operator(op1, site1, op2, site2, L)
    return np.vdot(psi, O @ psi)


def compute_expectation(psi: np.ndarray, L: int, op: np.ndarray, site: int) -> complex:
    """
    Compute single-site expectation <psi| O |psi>.

    Args:
        psi: State vector (2^L)
        L: Number of sites
        op: Single-site operator (2x2 matrix)
        site: Site index

    Returns:
        Complex expectation value
    """
    O = _single_site_operator(op, site, L)
    return np.vdot(psi, O @ psi)


def compute_zz_correlation_function(psi: np.ndarray, L: int,
                                    reference_site: Optional[int] = None) -> dict:
    """
    Compute ZZ correlation function C(r) = <Z_0 Z_r> - <Z_0><Z_r>.

    Args:
        psi: Ground state vector
        L: Number of sites
        reference_site: Reference site index (default: L//4 to reduce boundary effects)

    Returns:
        Dictionary with distances, raw ZZ correlator, and connected correlator
    """
    if reference_site is None:
        reference_site = L // 4

    i0 = reference_site
    z_i0 = np.real(compute_expectation(psi, L, SIGMA_Z, i0))

    distances = []
    raw_zz = []
    connected = []

    for r in range(0, L // 2):
        j = i0 + r
        if j >= L:
            break

        zz = np.real(compute_correlator(psi, L, SIGMA_Z, i0, SIGMA_Z, j))
        z_j = np.real(compute_expectation(psi, L, SIGMA_Z, j))

        distances.append(r)
        raw_zz.append(zz)
        connected.append(zz - z_i0 * z_j)

    return {
        'reference_site': i0,
        'distances': distances,
        'raw_zz': raw_zz,
        'connected': connected,
        'z_reference': z_i0,
    }


def estimate_correlation_length(psi: np.ndarray, L: int) -> float:
    """
    Estimate correlation length from ZZ correlator decay.

    Fits <Z_0 Z_r> ~ exp(-r/ξ) for large r.

    Args:
        psi: Ground state vector
        L: Number of sites

    Returns:
        Estimated correlation length (may be np.inf for ordered phase)
    """
    corr = compute_zz_correlation_function(psi, L)

    # Use absolute values for fitting
    distances = []
    log_corr = []

    for r, c in zip(corr['distances'][1:], corr['connected'][1:]):
        if np.abs(c) > 1e-12:
            distances.append(r)
            log_corr.append(np.log(np.abs(c)))

    if len(distances) < 3:
        return np.nan

    distances = np.array(distances)
    log_corr = np.array(log_corr)

    # Linear fit: log|C| = a - r/ξ
    coeffs = np.polyfit(distances, log_corr, 1)

    if coeffs[0] >= 0:
        return np.inf  # No decay - critical or ordered

    return -1.0 / coeffs[0]
