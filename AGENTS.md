# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview
Julia package for isometric Projected Entangled Pair States (IsoPEPS) — a tensor network method for 2D quantum many-body systems on cylinder geometries. The package provides parameterized quantum circuit ansatze, variational optimization (sampling-based and exact), transfer matrix analysis, and observable computation.

## Skills
- [test-runner](skills/test-runner/SKILL.md) -- Run Julia tests with coverage reporting and detailed output
- [add-gate](skills/add-gate/SKILL.md) -- Add a new quantum gate implementation to the package
- [add-observable](skills/add-observable/SKILL.md) -- Add a new observable measurement to the package
- [benchmark](skills/benchmark/SKILL.md) -- Run performance benchmarks for tensor operations and quantum computations
- [optimize-workflow](skills/optimize-workflow/SKILL.md) -- Set up and run optimization workflows for quantum state training
- [visualize](skills/visualize/SKILL.md) -- Generate plots and visualizations for quantum states and results
- [release](skills/release/SKILL.md) -- Create a new package release with version management

## Commands
```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file (e.g., gates only)
julia --project=. -e 'using IsoPEPS, Test; @testset "gates" begin include("test/gates.jl") end'

# Start REPL with project
julia --project=.

# Run a simulation script
julia --project=. project/simulation.jl
```

## Architecture

### Pipeline Flow
The core workflow is: **Parameters → Gates → Transfer Matrix / Quantum Channel → Observables → Optimization loop**.

1. **Parameters** (`Float64` vector) define rotation angles for the circuit ansatz
2. **Gates** (`src/gates.jl`) — `build_unitary_gate(params, p, row, nqubits)` constructs Rx-Rz brick-wall circuits. For 2×2 unit cells, `build_unitary_gate_2x2` produces alternating (odd, even) gate sets
3. **Two evaluation paths** for computing observables from gates:
   - **Exact**: `TransferOperator` (`src/transfer_matrix.jl`) contracts the tensor network exactly. `compute_transfer_spectrum` returns the dominant eigenvalue (energy) and spectral gap (correlation length). Observables computed in `src/observables_exact.jl`
   - **Sampling**: `sample_quantum_channel` (`src/quantum_channels.jl`) runs Yao circuits to collect Z/X/Y measurement samples. Observables computed in `src/observables_sampling.jl`
4. **Optimization** (`src/training.jl`) — `optimize_exact` (exact contraction + Optim/CMA-ES) or `optimize_circuit` (sampling-based + CMA-ES) minimize the energy

### Model System
`AbstractModel` (`src/models/abstract.jl`) defines the interface. Implementations:
- `TFIM(J, g)` — transverse-field Ising model: H = -g ΣXᵢ - J ΣZᵢZⱼ
- `HeisenbergJ1J2(J1, J2)` — J1-J2 Heisenberg model with 2×2 unit cell, requires Y measurements

Each model implements `compute_energy_from_samples`, `model_name`, and optionally `needs_y_measurement` and `default_unit_cell`.

### Transfer Matrix
`TransferOperator` (`src/transfer_matrix.jl`) unifies single-column (1×1) and multi-column (2×2) unit cells. Eigensolving is adaptive:
- Matrix size > 1024: matrix-free KrylovKit
- Matrix size > 256: iterative KrylovKit on explicit matrix
- Otherwise: full eigendecomposition

### Dual Observable System
Both `observables_exact.jl` and `observables_sampling.jl` export the same function names with different dispatch:
- `expect`, `correlation_function`, `structure_factor` — dispatch on `TransferOperator` (exact) vs `Vector{Float64}` samples
- `spin_spin_correlation`, `dimer_dimer_correlation`, `plaquette_plaquette_correlation` and their `*_structure_factor` counterparts

### Extensions (in `ext/`)
Optional heavy dependencies loaded via Julia's package extension system:
- `MPSKitExt` — VUMPS ground state reference via MPSKit/TensorKit
- `ITensorsExt` / `DMRGReferenceExt` — DMRG reference calculations
- `PEPSKitExt` — PEPSKit ground state reference
- `ManifoldsManoptExt` — particle swarm on unitary manifolds

Extension functions are stubbed in `src/IsoPEPS.jl` and populated when weak dependencies load.

### Results & I/O
`src/results_io.jl` — `save_result`/`load_result` serialize optimization results (3 result types: `CircuitOptimizationResult`, `ExactOptimizationResult`, `ManifoldOptimizationResult`) to JSON. `reconstruct_gates` rebuilds gates from saved parameters. `embed_params` warm-starts across system sizes.

### Simulation Scripts
`project/` contains end-to-end workflows:
- `simulation.jl` — parameter scanning with warm-start across scan values
- `postprocess.jl` — analysis and plotting of saved results
- `dmrg_reference.jl` — DMRG reference calculations for comparison

## Git Safety
- **NEVER force push** (`git push --force`, `git push -f`, `git push --force-with-lease`)
- Always create new commits rather than amending published commits
- Never skip hooks (--no-verify) unless explicitly requested

## Conventions
- 4-space indentation, Julia standard style
- Module files: lowercase with underscores (`transfer_matrix.jl`)
- Tests organized as `@testset` blocks in `test/runtests.jl`, one file per module
- Tensor contractions use OMEinsum with `GreedyMethod()` optimizer by default
- Quantum gates built with Yao framework (Rx, Rz rotations + CNOT entangling layers)
- Main branch: `master`
