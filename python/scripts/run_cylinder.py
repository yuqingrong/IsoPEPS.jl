#!/usr/bin/env python3
"""
Run TFIM cylinder benchmark with TenPy iDMRG.

Computes ground state energy for cylinders with circumference Ly = 1 to 4.

Usage:
    python scripts/run_cylinder.py --Ly 1 2 3 4 --chi 2 4 8 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfim.exact import exact_energy
from tfim.dmrg import run_idmrg_cylinder, check_tenpy
from tfim.utils import save_results_json, print_benchmark_header


def run_cylinder_benchmark(Ly_values: list[int], chi_values: list[int],
                           g: float, max_sweeps: int = 50) -> dict:
    """Run benchmark for different cylinder circumferences."""
    results = {
        'g': g,
        'description': f'TFIM cylinder benchmark at g={g}',
        'chi_values': chi_values,
        'Ly_values': Ly_values,
        'data': []
    }

    print_benchmark_header(f"TFIM Cylinder Benchmark (g = {g})")

    for Ly in Ly_values:
        print(f"\n--- Ly = {Ly} ---")

        for chi in chi_values:
            print(f"  chi = {chi}...", end=" ", flush=True)

            try:
                result = run_idmrg_cylinder(Ly, g, chi, max_sweeps=max_sweeps)
                E = result.energy
                xi = result.correlation_length
                n = result.n_sweeps

                xi_str = f"{xi:.2f}" if xi else "N/A"
                print(f"E/site = {E:.8f}, ξ = {xi_str}, sweeps = {n}")

                results['data'].append({
                    'Ly': Ly,
                    'chi_max': chi,
                    'energy_per_site': E,
                    'correlation_length': xi,
                    'n_sweeps': n,
                    'energy_history': result.energy_history,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                results['data'].append({
                    'Ly': Ly,
                    'chi_max': chi,
                    'energy_per_site': None,
                    'error': str(e),
                })

    # Add 1D exact reference
    results['exact_1d'] = exact_energy(g)
    print(f"\nExact 1D (Ly=1, Lx→∞): E/site = {results['exact_1d']:.10f}")

    return results


def plot_training_curves(results: dict, save_path: str = "training_curves.png"):
    """Plot energy convergence curves."""
    Ly_values = sorted(set(d['Ly'] for d in results['data'] if 'Ly' in d))

    n_plots = min(len(Ly_values), 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {2: 'blue', 4: 'red', 8: 'green', 16: 'purple', 32: 'orange'}

    for idx, Ly in enumerate(Ly_values[:4]):
        ax = axes[idx]

        Ly_data = [d for d in results['data']
                   if d.get('Ly') == Ly and d.get('energy_history')]

        if not Ly_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Ly = {Ly}')
            continue

        E_best = min(d['energy_per_site'] for d in Ly_data if d.get('energy_per_site'))

        for d in Ly_data:
            chi = d.get('chi_max', 0)
            history = d['energy_history']
            sweeps = range(1, len(history) + 1)

            color = colors.get(chi, 'gray')
            ax.plot(sweeps, history, '-', color=color, label=f'D={chi}', linewidth=2)

        ax.axhline(y=E_best, color='black', linestyle=':', linewidth=1.5,
                  label=f'Best: {E_best:.6f}')

        if Ly == 1:
            E_exact = results.get('exact_1d', exact_energy(results['g']))
            ax.axhline(y=E_exact, color='gray', linestyle='--', linewidth=1,
                      label=f'Exact: {E_exact:.6f}')

        ax.set_xlabel('DMRG Sweep', fontsize=11)
        ax.set_ylabel('Energy per site', fontsize=11)
        ax.set_title(f'Ly = {Ly} (cylinder circumference)', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'TFIM Training Curves at g = {results["g"]}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.show()


def print_summary_table(results: dict):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print(f"SUMMARY: TFIM Ground State Energy per site (g = {results['g']})")
    print("=" * 80)

    Ly_values = sorted(set(d.get('Ly') for d in results['data'] if d.get('Ly')))
    chi_values = sorted(set(d.get('chi_max', 0) for d in results['data']
                           if d.get('energy_per_site')))

    header = f"{'Ly':>4}"
    for chi in chi_values:
        header += f" |  D={chi:>2}  "
    print(header)
    print("-" * len(header))

    for Ly in Ly_values:
        row = f"{Ly:>4}"
        for chi in chi_values:
            d = next((x for x in results['data']
                     if x.get('Ly') == Ly and x.get('chi_max') == chi), None)
            if d and d.get('energy_per_site'):
                E = d['energy_per_site']
                row += f" | {E:>7.4f}"
            else:
                row += f" |    -   "
        print(row)

    print("-" * len(header))
    print(f"\n1D Analytical (Pfeuty): E/site = {results.get('exact_1d', 'N/A'):.10f}")


def main():
    parser = argparse.ArgumentParser(description='TFIM cylinder benchmark')
    parser.add_argument('--g', type=float, default=1.0, help='Transverse field strength')
    parser.add_argument('--chi', type=int, nargs='+', default=[2, 4, 8, 16],
                        help='Bond dimensions to test')
    parser.add_argument('--Ly', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Cylinder circumferences')
    parser.add_argument('--sweeps', type=int, default=50, help='Max DMRG sweeps')
    parser.add_argument('--output', type=str, default='cylinder_benchmark.json',
                        help='Output JSON file')
    args = parser.parse_args()

    if not check_tenpy():
        print("Error: TenPy is not installed. Install with: uv pip install physics-tenpy")
        sys.exit(1)

    results = run_cylinder_benchmark(args.Ly, args.chi, args.g, args.sweeps)

    # Save results
    save_results_json(results, args.output, description=f"TFIM cylinder benchmark g={args.g}")
    print(f"\nResults saved to {args.output}")

    # Print summary
    print_summary_table(results)

    # Plot training curves
    plot_training_curves(results)


if __name__ == "__main__":
    main()
