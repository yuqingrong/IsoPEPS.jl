"""
TFIM: Transverse Field Ising Model benchmark package.

Provides exact solutions, exact diagonalization, and DMRG calculations
for the 1D and quasi-1D (cylinder) TFIM.

Hamiltonian: H = -J Σ Z_i Z_j - g Σ X_i

References:
    [1] P. Pfeuty, Ann. Phys. 57, 79 (1970) - Exact 1D solution
    [2] TenPy: https://tenpy.readthedocs.io/
"""

from .exact import (
    exact_energy,
    exact_correlation_length,
    critical_point,
    magnetization_z,
    energy_gap,
    phase,
    format_xi,
)
from .ed import (
    build_hamiltonian,
    exact_diagonalization,
    compute_correlator,
    compute_expectation,
    compute_zz_correlation_function,
    estimate_correlation_length,
    EDResult,
)
from .constants import SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY

__version__ = "0.1.0"

__all__ = [
    # Exact analytical
    "exact_energy",
    "exact_correlation_length",
    "critical_point",
    "magnetization_z",
    "energy_gap",
    "phase",
    "format_xi",
    # Exact diagonalization
    "build_hamiltonian",
    "exact_diagonalization",
    "compute_correlator",
    "compute_expectation",
    "compute_zz_correlation_function",
    "estimate_correlation_length",
    "EDResult",
    # Constants
    "SIGMA_X",
    "SIGMA_Y",
    "SIGMA_Z",
    "IDENTITY",
]
