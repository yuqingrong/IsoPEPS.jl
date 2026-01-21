#!/usr/bin/env python3
"""
Compute and plot ZZ correlation functions for 1D TFIM.

Uses exact diagonalization for small systems.

Usage:
    python scripts/run_correlation.py --L 14 --g 0.5 0.8 1.0 1.5 2.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfim.exact import exact_correlation_length, format_xi
from tfim.ed import exact_diagonalization, compute_zz_correlation_function
from tfim.utils import save_results_json


def run_correlation_analysis(g_values: list[float], L: int) -> dict:
    """Compute correlation functions for multiple g values."""
    all_results = {
        'description': '1D TFIM ZZ correlation function',
        'hamiltonian': 'H = -J Σ Z_i Z_{i+1} - g Σ X_i',
        'system_size': L,
        'boundary_conditions': 'open',
        'results': []
    }

    for g in g_values:
        print(f"Computing correlation function for g={g}, L={L}...")

        ed = exact_diagonalization(L, g)
        corr = compute_zz_correlation_function(ed.ground_state, L)

        result = {
            'g': g,
            'L': L,
            'theoretical_xi': exact_correlation_length(g),
            'energy_per_site': ed.energy_per_site,
            'gap': ed.gap,
            **corr
        }
        all_results['results'].append(result)

    return all_results


def plot_correlations(data: dict, save_path: str = "correlation_plot.png"):
    """Plot correlation functions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(data['results'])))

    # Left plot: Log scale |C(r)|
    ax1 = axes[0]
    for i, result in enumerate(data['results']):
        g = result['g']
        r = np.array(result['distances'])
        C = np.array(result['connected'])

        mask = (r > 0) & (np.abs(C) > 1e-15)
        if np.sum(mask) > 0:
            ax1.semilogy(r[mask], np.abs(C[mask]), 'o-',
                        color=colors[i], label=f'g={g:.1f}', markersize=5)

            xi = result['theoretical_xi']
            if np.isfinite(xi) and xi < data['system_size']:
                r_theory = np.linspace(1, max(r[mask]), 50)
                C_theory = np.abs(C[1]) * np.exp(-r_theory / xi)
                ax1.semilogy(r_theory, C_theory, '--', color=colors[i], alpha=0.5)

    ax1.set_xlabel('Distance r', fontsize=12)
    ax1.set_ylabel('|C(r)| = |⟨Z₀Zᵣ⟩ - ⟨Z₀⟩⟨Zᵣ⟩|', fontsize=12)
    ax1.set_title(f'Connected ZZ Correlation (L={data["system_size"]})', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-8, 2)

    # Right plot: Raw <Z_i Z_j>
    ax2 = axes[1]
    for i, result in enumerate(data['results']):
        g = result['g']
        r = np.array(result['distances'])
        zz = np.array(result['raw_zz'])
        ax2.plot(r, zz, 'o-', color=colors[i], label=f'g={g:.1f}', markersize=5)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance r', fontsize=12)
    ax2.set_ylabel('⟨Z₀Zᵣ⟩', fontsize=12)
    ax2.set_title('Raw ZZ Correlation', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def print_correlation_table(data: dict):
    """Print correlation data as a table."""
    print("\n" + "=" * 80)
    print("CORRELATION FUNCTION SUMMARY")
    print("=" * 80)

    for result in data['results']:
        g = result['g']
        xi_theory = result['theoretical_xi']
        print(f"\ng = {g:.2f}  |  ξ_theory = {format_xi(xi_theory)}")
        print("-" * 50)
        print(f"{'r':>4}  {'⟨Z₀Zᵣ⟩':>12}  {'C(r) connected':>15}")
        print("-" * 50)

        for r, zz, c in zip(result['distances'][:7],
                            result['raw_zz'][:7],
                            result['connected'][:7]):
            print(f"{r:>4}  {zz:>12.6f}  {c:>15.6e}")


def main():
    parser = argparse.ArgumentParser(description='TFIM correlation function analysis')
    parser.add_argument('--L', type=int, default=14, help='System size (max ~18)')
    parser.add_argument('--g', type=float, nargs='+',
                        default=[0.5, 0.8, 0.9, 1.0, 1.1, 1.5, 2.0],
                        help='Transverse field values')
    parser.add_argument('--output', type=str, default='correlation_data.json',
                        help='Output JSON file')
    parser.add_argument('--plot', type=str, default='correlation_plot.png',
                        help='Output plot file')
    args = parser.parse_args()

    data = run_correlation_analysis(args.g, args.L)

    # Save to JSON
    save_results_json(data, args.output, description="1D TFIM ZZ correlation")
    print(f"\nData saved to {args.output}")

    # Print table
    print_correlation_table(data)

    # Plot
    plot_correlations(data, save_path=args.plot)


if __name__ == "__main__":
    main()
