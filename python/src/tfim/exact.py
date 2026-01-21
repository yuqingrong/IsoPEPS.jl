"""
Exact analytical solutions for the 1D Transverse Field Ising Model.

Hamiltonian: H = -J Σ Z_i Z_{i+1} - g Σ X_i

The 1D TFIM is exactly solvable via Jordan-Wigner transformation (Pfeuty 1970).
"""

import numpy as np
from scipy import integrate


# Critical point for 1D TFIM
CRITICAL_POINT_1D = 1.0

# Critical point for 2D TFIM (square lattice)
CRITICAL_POINT_2D = 3.044


def critical_point(dimension: int = 1) -> float:
    """
    Get the critical transverse field strength g_c for TFIM.

    Args:
        dimension: Lattice dimension (1 or 2)

    Returns:
        Critical field strength g_c = h_c/J
    """
    if dimension == 1:
        return CRITICAL_POINT_1D
    elif dimension == 2:
        return CRITICAL_POINT_2D
    else:
        raise ValueError(f"Unknown critical point for dimension {dimension}")


def exact_energy(g: float, J: float = 1.0) -> float:
    """
    Exact ground state energy per site for infinite 1D TFIM.

    E_0/N = -(1/π) ∫_0^π dk √(J² + g² - 2Jg cos(k))

    For the standard case J=1:
        E_0/N = -(1/π) ∫_0^π dk √(1 + g² - 2g cos(k))

    Args:
        g: Transverse field strength (h/J)
        J: Coupling strength (default 1.0)

    Returns:
        Ground state energy per site

    Reference:
        P. Pfeuty, Ann. Phys. 57, 79 (1970)
    """
    def integrand(k: float) -> float:
        return np.sqrt(J**2 + g**2 - 2*J*g*np.cos(k))

    result, _ = integrate.quad(integrand, 0, np.pi)
    return -result / np.pi


def exact_correlation_length(g: float) -> float:
    """
    Exact correlation length for infinite 1D TFIM at T=0.

    ξ = 1/|ln(g)|

    The correlation length diverges at the critical point g=1.

    Args:
        g: Transverse field strength (h/J)

    Returns:
        Correlation length in units of lattice spacing.
        Returns np.inf at the critical point g=1.

    Reference:
        P. Pfeuty, Ann. Phys. 57, 79 (1970)
    """
    if np.abs(g - CRITICAL_POINT_1D) < 1e-10:
        return np.inf
    return 1.0 / np.abs(np.log(g))


def magnetization_z(g: float, J: float = 1.0) -> float:
    """
    Exact longitudinal magnetization <Z> for 1D TFIM at T=0.

    For g < 1 (ordered phase): |<Z>| = (1 - g²)^(1/8)
    For g >= 1 (disordered phase): <Z> = 0

    Args:
        g: Transverse field strength
        J: Coupling strength (not used, for API consistency)

    Returns:
        Absolute value of longitudinal magnetization
    """
    if g >= CRITICAL_POINT_1D:
        return 0.0
    return (1 - g**2) ** 0.125


def energy_gap(g: float, J: float = 1.0) -> float:
    """
    Exact energy gap for infinite 1D TFIM.

    Δ = 2|J - g| = 2|1 - g| for J=1

    The gap closes at the critical point g=1.

    Args:
        g: Transverse field strength
        J: Coupling strength

    Returns:
        Energy gap
    """
    return 2 * np.abs(J - g)


def phase(g: float) -> str:
    """
    Determine the phase of the 1D TFIM.

    Args:
        g: Transverse field strength

    Returns:
        Phase name: "ordered", "critical", or "disordered"
    """
    if np.abs(g - CRITICAL_POINT_1D) < 1e-10:
        return "critical"
    elif g < CRITICAL_POINT_1D:
        return "ordered"
    else:
        return "disordered"


def format_xi(xi: float) -> str:
    """Format correlation length for display (handles infinity)."""
    if np.isfinite(xi):
        return f"{xi:.4f}"
    return "∞"
