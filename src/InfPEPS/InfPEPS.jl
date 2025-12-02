"""
    InfPEPS

A submodule for infinite Projected Entangled Pair States (InfPEPS) algorithms,
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
module InfPEPS

using LinearAlgebra
using Statistics
using Yao, Yao.EasyBuild
using OMEinsum
using OMEinsumContractionOrders
using ITensors,TensorKit, MPSKit, MPSKitModels, PEPSKit
using MPSKitModels: transverse_field_ising, InfiniteStrip, InfiniteCylinder, @mpoham, InfiniteChain, nearest_neighbours, vertices
using PEPSKit: InfiniteSquare
using Optimization, OptimizationCMAEvolutionStrategy, OptimizationOptimisers
using Manifolds, Manopt
using Optim
using Plots
using Colors: RGBA
using BinningAnalysis, LsqFit

# Import parent module utilities
import ..IsoPEPS: IndexStore, newindex!

# Include submodule files
include("quantum_channels.jl")
include("gate_and_cost.jl")
include("training.jl")
include("visualization.jl")
include("refer.jl")
include("exact.jl")
# Re-export main functions
export iterate_channel_PEPS, iterate_dm
# exact
export contract_Elist, exact_left_eigen, single_transfer
export cost_X, cost_ZZ, cost_singleop, cost_ZZ_single, cost_X_circ, cost_ZZ_circ
export build_gate_from_params, energy_measure
export train_energy_circ, train_exact, train_energy_circ_gradient, train_hybrid, train_nocompile
export check_gap_sensitivity, check_all_gap_sensitivity_combined
# visualization
export block_variance, dynamics_observables, dynamics_observables_all, eigenvalues,gap, ACF, correlation, energy_con
export save_training_data
export exact_energy_PEPS, exact_iPEPS
# refer.jl
export result_MPSKit, result_PEPSKit, result_1d
end 

