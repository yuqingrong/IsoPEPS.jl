# InfPEPS Submodule

A modular implementation of infinite Projected Entangled Pair States (InfPEPS) algorithms for quantum many-body physics.

## Structure

The InfPEPS submodule is organized into focused components:

### Core Modules

```
InfPEPS/
├── InfPEPS.jl                # Main submodule file
├── quantum_channels.jl       # Quantum channel iteration
├── transfer_matrices.jl      # Tensor network contractions
├── cost_functions.jl         # Observable expectation values
├── gate_construction.jl      # Parameterized quantum gates
├── training.jl               # Variational optimization
├── analysis.jl               # Sensitivity analysis
├── visualization.jl          # Plotting and visualization
├── data_io.jl               # Data saving/loading
└── benchmarks.jl            # Reference implementations
```

## Usage

### Basic Import

```julia
using IsoPEPS
using IsoPEPS.InfPEPS  # Access submodule directly
```

### Backward Compatibility

All functions are re-exported at the top level for backward compatibility:

```julia
using IsoPEPS

# These work as before
rho, Z_list, X_list = iterate_channel_PEPS(gate, row)
energy = cost_X(rho, row, gate)
train_energy_circ(params, J, g, p, row)
```

### Direct Submodule Access

For better organization, you can access submodule components directly:

```julia
using IsoPEPS.InfPEPS

# Quantum channels
rho, gap = InfPEPS.exact_left_eigen(gate, nsites)

# Training
results = InfPEPS.train_energy_circ(params, J, g, p, row)

# Analysis
InfPEPS.check_gap_sensitivity(params, idx, g, row, p)

# Visualization
InfPEPS.draw_gap()
```

## Module Organization

### quantum_channels.jl
- `iterate_channel_PEPS`: Iterate quantum channel with measurements
- `exact_left_eigen`: Compute spectral gap and fixed points

### transfer_matrices.jl
- `contract_Elist`: Efficient tensor network contraction for transfer matrices

### cost_functions.jl
- `cost_X`, `cost_ZZ`: Observable expectation values (exact)
- `cost_X_circ`, `cost_ZZ_circ`: Observable expectation values (sampling)
- `exact_energy_PEPS`: Benchmark using MPSKit

### gate_construction.jl
- `build_gate_from_params`: Construct unitary gates from parameters
- `build_parameterized_gate`: Single layer gate construction
- `extract_Z_configurations`: Parse measurement data
- `compute_energy_from_measurements`: Energy from sampling

### training.jl
- `train_energy_circ`: Variational optimization with CMA-ES
- `train_nocompile`: Manifold-based optimization

### analysis.jl
- `check_gap_sensitivity`: Single parameter sensitivity analysis
- `check_all_gap_sensitivity_combined`: Global sensitivity analysis

### visualization.jl
- `draw_X_from_file`: Convergence analysis plots
- `draw_gap`: Spectral gap vs field strength
- `draw`: Benchmark comparison plots

### data_io.jl
- `save_training_data`: Save measurement and gap data

### benchmarks.jl
- `exact_iPEPS`: PEPSKit reference implementation

## Example Workflow

```julia
using IsoPEPS
using IsoPEPS.InfPEPS

# 1. Setup parameters
p = 2  # Number of layers
row = 3  # Number of rows
g = 1.0  # Transverse field
J = 1.0  # Coupling
params = rand(6*p) * 2π

# 2. Train variational circuit
X_history, final_A, final_params, cost, Z_lists, X_lists, gaps, param_hist = 
    train_energy_circ(params, J, g, p, row; maxiter=1000)

# 3. Analyze sensitivity
check_gap_sensitivity(final_params, 1, g, row, p; 
                     save_path="sensitivity_param1.pdf")

# 4. Check convergence
draw_X_from_file(g, [1, 10, 100]; save_path="convergence.pdf")

# 5. Save results
save_training_data(Z_lists, X_lists, gaps)

# 6. Compare with benchmarks
exact_energy = exact_iPEPS(2, 4, J, g)
println("Final energy: $cost, Exact: $exact_energy")
```

## Design Principles

### 1. **Modularity**
- Each file handles one specific concern
- Functions are focused and composable
- Clear separation of computation, I/O, and visualization

### 2. **Backward Compatibility**
- All existing code continues to work
- Functions re-exported at top level
- No breaking changes

### 3. **Documentation**
- Every function has docstrings
- Clear argument descriptions
- Usage examples included

### 4. **Testability**
- Small, focused functions easy to test
- Clear input/output contracts
- Minimal side effects

## Benefits of Modular Structure

1. **Easier Maintenance**: Find and fix bugs quickly
2. **Better Testing**: Test each component independently
3. **Improved Readability**: Understand code organization at a glance
4. **Flexible Usage**: Import only what you need
5. **Parallel Development**: Work on different modules simultaneously

## Migration Guide

No changes needed! Your existing code works as-is. But for new code, consider:

**Old style:**
```julia
using IsoPEPS
result = train_energy_circ(params, J, g, p, row)
```

**New style (clearer namespacing):**
```julia
using IsoPEPS
using IsoPEPS.InfPEPS
result = InfPEPS.train_energy_circ(params, J, g, p, row)
```

Both work identically!

