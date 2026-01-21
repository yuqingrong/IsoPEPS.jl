#!/usr/bin/env python3
"""
Run 1D TFIM benchmark with TenPy iDMRG.

Usage:
    python scripts/run_benchmark.py --chi 2 4 8 16 32
    python scripts/run_benchmark.py --g 0.5 0.8 1.0 1.5 2.0
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfim.exact import exact_energy, exact_correlation_length, format_xi
from tfim.dmrg import run_idmrg_1d, check_tenpy
from tfim.utils import save_results_json, print_benchmark_header, create_benchmark_plot


def run_benchmark(g_values: list[float], chi_values: list[int],
                  max_sweeps: int = 100) -> list[dict]:
    """Run benchmark for multiple g and chi values."""
    results = []

    print_benchmark_header("1D TFIM Benchmark with TenPy iDMRG")

    for g in g_values:
        E_exact = exact_energy(g)
        xi_exact = exact_correlation_length(g)

        print(f"\ng = {g} | E_exact = {E_exact:.8f} | ξ_exact = {format_xi(xi_exact)}")
        print("-" * 60)

        for chi in chi_values:
            print(f"  D={chi:>3}...", end=" ", flush=True)

            try:
                result = run_idmrg_1d(g, chi, max_sweeps=max_sweeps)
                E = result.energy
                xi = result.correlation_length
                n = result.n_sweeps
                err = abs(E - E_exact)

                xi_str = f"{xi:.2f}" if xi else "N/A"
                print(f"E = {E:.8f} (err={err:.2e}), ξ = {xi_str}, sweeps={n}")

                results.append({
                    'g': g,
                    'chi_max': chi,
                    'energy': E,
                    'energy_exact': E_exact,
                    'xi': xi,
                    'xi_exact': xi_exact,
                    'n_sweeps': n,
                    'energy_history': result.energy_history,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    'g': g,
                    'chi_max': chi,
                    'energy': None,
                    'error': str(e),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description='1D TFIM benchmark with iDMRG')
    parser.add_argument('--g', type=float, nargs='+',
                        default=[0.5, 0.8, 0.9, 1.0, 1.1, 1.5, 2.0],
                        help='Transverse field values')
    parser.add_argument('--chi', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                        help='Bond dimensions to test')
    parser.add_argument('--sweeps', type=int, default=20, help='Max DMRG sweeps')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()

    if not check_tenpy():
        print("Error: TenPy is not installed. Install with: uv pip install physics-tenpy")
        sys.exit(1)

    results = run_benchmark(args.g, args.chi, args.sweeps)

    # Save results
    save_results_json(results, args.output, description="1D TFIM iDMRG benchmark")
    print(f"\nResults saved to {args.output}")

    # Plot if requested
    if args.plot:
        # Get best result for each g (highest chi)
        best_results = {}
        for r in results:
            if r.get('energy') is not None:
                g = r['g']
                if g not in best_results or r['chi_max'] > best_results[g]['chi_max']:
                    best_results[g] = r
        create_benchmark_plot(list(best_results.values()), save_path='benchmark_plot.png')


if __name__ == "__main__":
    main()
