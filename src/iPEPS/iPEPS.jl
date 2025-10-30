"""
    iPEPS

A submodule for infinite Projected Entangled Pair States (iPEPS) algorithms,
providing functionality for:
- Quantum channel simulations
- Transfer matrix calculations
- Cost function evaluations
- Variational optimization
- Sensitivity analysis
- Data visualization and I/O

# Submodules
- `QuantumChannels`: Channel iteration and spectral gap calculations
- `TransferMatrices`: Tensor network contractions for transfer matrices
- `CostFunctions`: Energy and observable cost functions
- `Training`: Variational optimization routines
- `Analysis`: Sensitivity analysis tools
- `Visualization`: Plotting and data visualization
- `DataIO`: Data saving and loading utilities
"""
module iPEPS

using LinearAlgebra
using Statistics
using Yao, Yao.EasyBuild
using OMEinsum
using OMEinsumContractionOrders
using TensorKit, MPSKit, MPSKitModels, PEPSKit
using MPSKitModels: transverse_field_ising, InfiniteStrip, InfiniteCylinder
using PEPSKit: InfiniteSquare
using Optimization, OptimizationCMAEvolutionStrategy
using Manifolds, Manopt
using Plots
using Colors: RGBA

# Import parent module utilities
import ..IsoPEPS: IndexStore, newindex!

# Include submodule files
include("quantum_channels.jl")
include("transfer_matrices.jl")
include("cost_functions.jl")
include("gate_construction.jl")
include("training.jl")
include("analysis.jl")
include("visualization.jl")
include("data_io.jl")
include("benchmarks.jl")

# Re-export main functions
export iterate_channel_PEPS, exact_left_eigen
export contract_Elist
export cost_X, cost_ZZ, cost_X_circ, cost_ZZ_circ
export build_gate_from_params, build_parameterized_gate
export train_energy_circ, train_nocompile
export check_gap_sensitivity, check_all_gap_sensitivity_combined
export draw_X_from_file, draw_gap, draw
export save_training_data
export exact_energy_PEPS, exact_iPEPS

end # module iPEPS

