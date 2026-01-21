# TFIM Benchmarks

Benchmark calculations for the Transverse Field Ising Model (TFIM) ground state energy and correlation length.

## Installation

```bash
# Using uv
uv sync

# With TenPy support (for DMRG calculations)
uv sync --extra tenpy

# For development
uv sync --extra dev
```

## Package Structure

```
python/
├── src/tfim/           # Core package
│   ├── exact.py        # Exact analytical formulas (Pfeuty 1970)
│   ├── ed.py           # Exact diagonalization for small systems
│   ├── dmrg.py         # TenPy DMRG wrappers
│   ├── constants.py    # Pauli matrices and physical constants
│   └── utils.py        # I/O and plotting utilities
├── scripts/            # Command-line scripts
│   ├── run_benchmark.py
│   ├── run_correlation.py
│   └── run_cylinder.py
└── tests/              # pytest test suite
```

## Hamiltonian

The TFIM Hamiltonian is:

```
H = -J Σ Z_i Z_j - g Σ X_i
```

where:
- `J` is the ferromagnetic coupling (default: 1.0)
- `g = h/J` is the dimensionless transverse field strength
- `g_c = 1.0` is the critical point (1D)

## Usage

### Exact Analytical Results

```python
from tfim import exact_energy, exact_correlation_length

# Ground state energy per site
E = exact_energy(g=1.0)  # ≈ -1.2732 at critical point

# Correlation length
xi = exact_correlation_length(g=0.5)  # = 1/|ln(0.5)| ≈ 1.44
```

### Exact Diagonalization

```python
from tfim import exact_diagonalization, compute_zz_correlation_function

# Diagonalize 12-site system
result = exact_diagonalization(L=12, g=1.0)
print(f"E/N = {result.energy_per_site}")

# Compute correlation function
corr = compute_zz_correlation_function(result.ground_state, L=12)
```

### DMRG Calculations (requires TenPy)

```python
from tfim.dmrg import run_idmrg_1d

result = run_idmrg_1d(g=1.0, chi_max=32)
print(f"E = {result.energy}, ξ = {result.correlation_length}")
```

## Command-Line Scripts

```bash
# Run 1D benchmark
python scripts/run_benchmark.py --g 0.5 1.0 2.0 --chi 2 4 8 16

# Compute correlation functions
python scripts/run_correlation.py --L 14 --g 0.5 1.0 2.0

# Cylinder benchmark
python scripts/run_cylinder.py --Ly 1 2 3 4 --chi 2 4 8
```

## Running Tests

```bash
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src/tfim --cov-report=term-missing
```

## References

1. P. Pfeuty, "The one-dimensional Ising model with a transverse field", Ann. Phys. 57, 79 (1970)
2. TenPy: https://tenpy.readthedocs.io/
