"""
DMRG calculations for TFIM using TenPy.

Provides both infinite DMRG (iDMRG) and finite DMRG implementations.
"""

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np


@dataclass
class DMRGResult:
    """Result from DMRG calculation."""
    g: float
    J: float
    chi_max: int
    energy: float
    energy_history: list[float] = field(default_factory=list)
    correlation_length: Optional[float] = None
    n_sweeps: int = 0
    method: str = "iDMRG"
    Ly: int = 1  # Cylinder circumference (1 = chain)


def check_tenpy():
    """Check if TenPy is available."""
    try:
        import tenpy
        return True
    except ImportError:
        return False


def run_idmrg_1d(g: float, chi_max: int, J: float = 1.0,
                 max_sweeps: int = 100, conv_tol: float = 1e-12) -> DMRGResult:
    """
    Run infinite DMRG for 1D TFIM chain.

    Args:
        g: Transverse field strength
        chi_max: Maximum bond dimension
        J: Coupling strength
        max_sweeps: Maximum number of DMRG sweeps
        conv_tol: Energy convergence tolerance

    Returns:
        DMRGResult with energy and correlation length
    """
    from tenpy.models.tf_ising import TFIChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg

    model_params = {
        'L': 2,  # Unit cell
        'J': J,
        'g': g,
        'bc_MPS': 'infinite',
        'conserve': None,
    }

    model = TFIChain(model_params)
    psi = MPS.from_lat_product_state(model.lat, [['up']])

    dmrg_params = {
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-12,
        },
    }

    energy_history = []

    for sweep in range(max_sweeps):
        info = dmrg.run(psi, model, dmrg_params)
        E = info['E']
        energy_history.append(float(E))

        # Check convergence
        if sweep > 5 and len(energy_history) > 1:
            if abs(energy_history[-1] - energy_history[-2]) < conv_tol:
                break

    # Get correlation length (suppress deprecation warning)
    xi = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            xi = float(psi.correlation_length())
    except Exception:
        xi = None

    return DMRGResult(
        g=g,
        J=J,
        chi_max=chi_max,
        energy=energy_history[-1] if energy_history else np.nan,
        energy_history=energy_history,
        correlation_length=xi if xi and np.isfinite(xi) else None,
        n_sweeps=len(energy_history),
        method="iDMRG",
        Ly=1,
    )


def run_idmrg_cylinder(Ly: int, g: float, chi_max: int, J: float = 1.0,
                       max_sweeps: int = 50, conv_tol: float = 1e-10) -> DMRGResult:
    """
    Run infinite DMRG for TFIM on cylinder with circumference Ly.

    Args:
        Ly: Cylinder circumference (number of sites in y-direction)
        g: Transverse field strength
        chi_max: Maximum bond dimension
        J: Coupling strength
        max_sweeps: Maximum number of sweeps
        conv_tol: Energy convergence tolerance

    Returns:
        DMRGResult with energy per site
    """
    if Ly == 1:
        return run_idmrg_1d(g, chi_max, J, max_sweeps, conv_tol)

    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    from tenpy.models.lattice import Square
    from tenpy.algorithms import dmrg

    class TFICylinder(CouplingMPOModel):
        """TFIM on infinite cylinder."""

        def init_sites(self, model_params):
            conserve = model_params.get('conserve', None)
            return SpinHalfSite(conserve=conserve)

        def init_lattice(self, model_params):
            Ly = model_params.get('Ly', 2)
            bc_MPS = model_params.get('bc_MPS', 'infinite')
            site = self.init_sites(model_params)
            return Square(Lx=1, Ly=Ly, site=site, bc_MPS=bc_MPS, bc=['periodic', 'periodic'])

        def init_terms(self, model_params):
            J = model_params.get('J', 1.)
            g = model_params.get('g', 1.)
            self.add_onsite(-g, 0, 'Sigmax')
            self.add_coupling(-J, 0, 'Sigmaz', 0, 'Sigmaz', [1, 0])
            self.add_coupling(-J, 0, 'Sigmaz', 0, 'Sigmaz', [0, 1])

    model_params = {
        'Ly': Ly,
        'J': J,
        'g': g,
        'bc_MPS': 'infinite',
        'conserve': None,
    }

    model = TFICylinder(model_params)
    psi = MPS.from_lat_product_state(model.lat, [['up']])

    dmrg_params = {
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-12,
        },
    }

    energy_history = []
    E_old = 0

    for sweep in range(max_sweeps):
        info = dmrg.run(psi, model, dmrg_params)
        E = info['E'] / Ly  # Energy per site
        energy_history.append(float(E))

        if abs(E - E_old) < conv_tol and sweep > 5:
            break
        E_old = E

    xi = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            xi = float(psi.correlation_length())
    except Exception:
        xi = None

    return DMRGResult(
        g=g,
        J=J,
        chi_max=chi_max,
        energy=energy_history[-1] if energy_history else np.nan,
        energy_history=energy_history,
        correlation_length=xi if xi and np.isfinite(xi) else None,
        n_sweeps=len(energy_history),
        method="iDMRG",
        Ly=Ly,
    )


def run_finite_dmrg(L: int, g: float, chi_max: int, J: float = 1.0,
                    max_sweeps: int = 50) -> DMRGResult:
    """
    Run finite DMRG for 1D TFIM chain.

    Args:
        L: System size
        g: Transverse field strength
        chi_max: Maximum bond dimension
        J: Coupling strength
        max_sweeps: Maximum sweeps

    Returns:
        DMRGResult with energy per site
    """
    from tenpy.models.tf_ising import TFIChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg

    model_params = {
        'L': L,
        'J': J,
        'g': g,
        'bc_MPS': 'finite',
        'conserve': None,
    }

    model = TFIChain(model_params)
    psi = MPS.from_lat_product_state(model.lat, [['up']])

    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10,
        },
        'max_sweeps': max_sweeps,
    }

    info = dmrg.run(psi, model, dmrg_params)
    energy_per_site = info['E'] / L

    return DMRGResult(
        g=g,
        J=J,
        chi_max=chi_max,
        energy=energy_per_site,
        energy_history=[energy_per_site],
        n_sweeps=info.get('sweeps', max_sweeps),
        method=f"finite DMRG (L={L})",
        Ly=1,
    )
