"""
    IsoPEPS

A Julia package for infinite Projected Entangled Pair States (IsoPEPS) algorithms,
providing functionality for:
- Quantum channel simulations
- Transfer matrix calculations  
- Cost function evaluations
- Variational optimization
- Sensitivity analysis
- Data visualization
"""
module IsoPEPS

using LinearAlgebra
using Statistics
using Yao, Yao.EasyBuild
using OMEinsum
using OMEinsumContractionOrders
using TensorKit, MPSKit, MPSKitModels, PEPSKit
using MPSKitModels: transverse_field_ising, InfiniteStrip, InfiniteCylinder, @mpoham, InfiniteChain, nearest_neighbours, vertices
using PEPSKit: InfiniteSquare
using Optimization, OptimizationCMAEvolutionStrategy
using Manifolds, Manopt
using Optim
using CairoMakie
using LsqFit
using JSON3

# Index helper for tensor contractions
mutable struct IndexStore
    count::Int
    IndexStore() = new(0)
end

function newindex!(store::IndexStore)
    store.count += 1
    return store.count
end

# Include module files
include("quantum_channels.jl")
include("gate_and_cost.jl")
include("training.jl")
include("visualization.jl")
include("refer.jl")
include("exact.jl")

# iter circuit
export iterate_channel_PEPS, iterate_dm

# exact
export contract_Elist, exact_left_eigen, single_transfer, exact_E_from_params
export cost_X, cost_ZZ, cost_singleop, cost_ZZ_single, cost_X_circ, cost_ZZ_circ

#gate & cost
export build_gate_from_params, energy_measure, energy_recal

#training
export train_energy_circ, train_exact, train_energy_circ_gradient, train_hybrid, train_nocompile

# visualization & data I/O
export TrainingData, save_data, load_data, save_results, load_results
export plot_correlation_heatmap, plot_acf, compute_acf, fit_acf_exponential
export plot_training_history, plot_variance_vs_samples, visualize_training

# refer.jl
export result_MPSKit, result_PEPSKit, result_1d

end