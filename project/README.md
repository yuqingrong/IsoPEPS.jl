# Research scripts

This directory contains research simulation, reference, and post-processing
scripts for IsoPEPS.

## Environment

Use the repository-root Julia environment for every script here. From the
repository root:

```bash
julia --project=. project/simulation.jl
julia --project=. project/postprocess.jl <target>
```

For interactive work, start `julia --project=.` from the repository root and
then include the desired script, for example
`include("project/simulation.jl")`.

`project/Project.toml` and `project/Manifest.toml` are retained only as legacy
records of an earlier standalone research environment. Do not update or
instantiate them for the current workflow.

## Contents

### Simulation Scripts

- **`simulation.jl`** - Main simulation driver for training and benchmarking IsoPEPS circuits
  - `simulation()` - Run single simulation for given parameters
  - `parallel_simulation_threaded()` - Run simulations in parallel for multiple transverse field values
  - `analyze_trained_gate()` - Load and analyze trained gates from data files

## Usage

```julia
# Start Julia from the repository root with `--project=.`; then include and run
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

Research outputs are local files under `project/results/`. They are not part
of the Julia package or the committed source tree. Use the reproducibility
workflow documented in the root README to create a curated paper-data package.
