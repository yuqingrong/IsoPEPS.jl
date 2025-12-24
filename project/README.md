# Research Scripts

This directory contains research code and simulation scripts for the IsoPEPS package.

## Setup

This folder has its own Julia environment. To use it:

```julia
cd("project")
using Pkg
Pkg.activate(".")
Pkg.instantiate()  # First time only
```

Or from the command line:
```bash
cd project
julia --project=.
```

## Contents

### Simulation Scripts

- **`simulation.jl`** - Main simulation driver for training and benchmarking IsoPEPS circuits
  - `simulation()` - Run single simulation for given parameters
  - `parallel_simulation_threaded()` - Run simulations in parallel for multiple transverse field values
  - `analyze_trained_gate()` - Load and analyze trained gates from data files

## Usage

```julia
# Activate the project environment
using Pkg
Pkg.activate("project")

# Include and run
include("project/simulation.jl")

# Or interactively use functions after including
J = 1.0  # Coupling strength
g = 1.0  # Transverse field
row = 3  # Number of rows
p = 3    # Number of layers
nqubits = 3  # Qubits per gate

simulation(J, g, row, p, nqubits; maxiter=1000)
```

## Output

Simulation results are saved to a `data/` directory with naming convention:
- `compile_energy_history_*.dat` - Energy convergence history
- `compile_params_history_*.dat` - Parameter evolution
- `compile_Z_list_list_*.dat` - Z measurement results
- `compile_X_list_list_*.dat` - X measurement results
