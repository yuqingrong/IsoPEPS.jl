"""
Utility functions for TFIM calculations.

Provides JSON I/O, plotting, and formatting utilities.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def save_results_json(results: dict | list, filename: str | Path,
                      description: str = "TFIM results") -> None:
    """
    Save results to JSON file, handling numpy types.

    Args:
        results: Results dictionary or list
        filename: Output file path
        description: Description to include in file
    """
    def convert_numpy(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer)):
            val = float(obj)
            if np.isfinite(val):
                return val
            return None
        elif isinstance(obj, float):
            if np.isfinite(obj):
                return obj
            return None
        elif isinstance(obj, np.ndarray):
            return [convert_numpy(x) for x in obj.tolist()]
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output = {
        'description': description,
        'hamiltonian': 'H = -J Σ Z_i Z_j - g Σ X_i',
    }

    if isinstance(results, list):
        output['results'] = convert_numpy(results)
    else:
        output.update(convert_numpy(results))

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)


def load_results_json(filename: str | Path) -> dict:
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def format_energy_error(E: float, E_exact: float) -> str:
    """Format energy error for display."""
    err = abs(E - E_exact)
    if err < 1e-14:
        return "< 1e-14"
    return f"{err:.2e}"


def format_correlation_length(xi: Optional[float]) -> str:
    """Format correlation length for display (handles infinity/None)."""
    if xi is None:
        return "N/A"
    if np.isfinite(xi):
        return f"{xi:.2f}"
    return "∞"


def print_benchmark_header(title: str, width: int = 80) -> None:
    """Print formatted benchmark header."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_benchmark_row(g: float, E: float, E_exact: float,
                        xi: Optional[float], xi_exact: float,
                        n_sweeps: Optional[int] = None) -> None:
    """Print formatted benchmark result row."""
    err = abs(E - E_exact)
    xi_str = format_correlation_length(xi)
    xi_exact_str = format_correlation_length(xi_exact)

    if n_sweeps is not None:
        print(f"  g={g:.2f}: E={E:.8f} (err={err:.2e}), "
              f"ξ={xi_str} (exact={xi_exact_str}), sweeps={n_sweeps}")
    else:
        print(f"  g={g:.2f}: E={E:.8f} (err={err:.2e}), "
              f"ξ={xi_str} (exact={xi_exact_str})")


def create_benchmark_plot(results: list[dict], save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    Create standard benchmark plots for TFIM results.

    Args:
        results: List of result dictionaries with keys:
                 g, energy, energy_exact, xi, xi_exact
        save_path: Path to save figure (optional)
        show: Whether to display the figure
    """
    import matplotlib.pyplot as plt

    g_vals = [r['g'] for r in results]
    e_dmrg = [r['energy'] for r in results]
    e_exact = [r['energy_exact'] for r in results]
    xi_dmrg = [r.get('xi') for r in results]
    xi_exact = [r['xi_exact'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy plot
    ax1 = axes[0]
    ax1.plot(g_vals, e_exact, 'b-', linewidth=2, label='Exact (Pfeuty)')
    ax1.plot(g_vals, e_dmrg, 'ro', markersize=8, label='DMRG')
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('g = h/J', fontsize=12)
    ax1.set_ylabel('Energy per site', fontsize=12)
    ax1.set_title('Ground State Energy', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Correlation length plot
    ax2 = axes[1]
    xi_exact_plot = [x if np.isfinite(x) else np.nan for x in xi_exact]
    xi_dmrg_plot = [x if x and np.isfinite(x) else np.nan for x in xi_dmrg]

    ax2.semilogy(g_vals, xi_exact_plot, 'b-', linewidth=2, label='Exact: ξ=1/|ln(g)|')
    valid_xi = [(g, x) for g, x in zip(g_vals, xi_dmrg_plot) if not np.isnan(x)]
    if valid_xi:
        ax2.semilogy([v[0] for v in valid_xi], [v[1] for v in valid_xi],
                     'ro', markersize=8, label='DMRG')
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('g = h/J', fontsize=12)
    ax2.set_ylabel('Correlation length ξ', fontsize=12)
    ax2.set_title('Correlation Length', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()


def create_training_curves_plot(results: list, title: str = "DMRG Training Curves",
                                save_path: Optional[str] = None,
                                show: bool = True) -> None:
    """
    Plot energy convergence (training) curves.

    Args:
        results: List of DMRGResult or dicts with energy_history
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, r in enumerate(results):
        if hasattr(r, 'energy_history'):
            history = r.energy_history
            label = f"g={r.g:.1f}, D={r.chi_max}"
        else:
            history = r.get('energy_history', [])
            label = f"g={r.get('g', '?')}, D={r.get('chi_max', '?')}"

        if history:
            sweeps = range(1, len(history) + 1)
            ax.plot(sweeps, history, '-o', color=colors[i], label=label, markersize=4)

    ax.set_xlabel('DMRG Sweep', fontsize=12)
    ax.set_ylabel('Energy per site', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    if show:
        plt.show()
