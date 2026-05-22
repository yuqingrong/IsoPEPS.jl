# IsoPEPS

[![Build Status](https://github.com/yuqingrong/IsoPEPS.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/yuqingrong/IsoPEPS.jl/actions/workflows/CI.yml?query=branch%3Amaster)

IsoPEPS is a Julia package for isometric Projected Entangled Pair States on
cylinder geometries. It provides parameterized quantum-circuit ansatze, exact
transfer-matrix contraction, sampling-based quantum-channel evaluation,
observable calculations, and variational optimization workflows for 2D quantum
many-body models.

## Features

- Brick-wall Rx/Rz circuit ansatze with CNOT entangling layers.
- Exact transfer-operator contraction and spectral analysis.
- Sampling-based observable estimation with Yao circuits.
- TFIM and J1-J2 Heisenberg model support.
- Optimization workflows for circuit and exact-contraction objectives.
- Optional reference calculations through package extensions for MPSKit,
  PEPSKit, ITensors, and Manopt.

## Installation

Clone the repository and instantiate the Julia environment:

```bash
git clone https://github.com/yuqingrong/IsoPEPS.jl.git
cd IsoPEPS.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

For day-to-day development, the Makefile wraps the common commands:

```bash
make init
make test
make test-gates
```

## Quick Start

```julia
using IsoPEPS
using Random

Random.seed!(1)

p = 2
row = 3
nqubits = 3
params = 2 * pi .* rand(gate_parameter_count(p, nqubits))

gates = build_unitary_gate(params, p, row, nqubits)
model = TFIM(J=1.0, g=2.0)

energy, x, zz_vertical, zz_horizontal =
    compute_exact_energy_from_gates(model, gates, row, nqubits)
```

## Testing

Run the full package test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run one named test group:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["gates"])'
make test-gates
make test-transfer_matrix
```

The available test groups are defined in `test/runtests.jl`.

## Repository Layout

- `src/`: package source code.
- `src/models/`: model interfaces and implementations.
- `ext/`: optional Julia package extensions for heavier reference workflows.
- `test/`: package tests, organized as named test groups.
- `project/`: research scripts with their own Julia environment.
- `simulations/`: configuration-driven simulation workflows.
- `note/` and `typst/`: research notes and manuscript/poster materials.

Generated outputs such as `data/`, `image/`, `states/`, `project/results/`,
and `simulations/results/` are intentionally ignored by Git.

## Research Scripts

The `project/` directory contains end-to-end research workflows for scans,
post-processing, and reference calculations. See `project/README.md` for setup
and usage details.

## Contributing

Please see `CONTRIBUTING.md` for the expected branch, testing, formatting, and
pull request workflow.

## License

This project is available under the MIT license. See `LICENSE`.
