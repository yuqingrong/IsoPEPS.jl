"""
    IsoPEPS

A Julia package for Isometric Projected Entangled Pair States (IsoPEPS) algorithms.

# Features
- Quantum channel simulation for PEPS
- Transfer matrix calculations with exact contraction
- Variational optimization (CMA-ES, manifold methods)
- Reference implementations using MPSKit and PEPSKit
- Data visualization and analysis

# Main Functions
- `sample_quantum_channel`: Sample observables from quantum channel
- `build_unitary_gate`: Build parameterized unitary gates
- `compute_energy`: Compute TFIM energy from samples
- `optimize_circuit`: Optimize circuit parameters
- `compute_transfer_spectrum`: Compute transfer matrix spectrum
- `mpskit_ground_state`: Reference ground state from MPSKit
"""
module IsoPEPS

using LinearAlgebra
using Statistics
using KrylovKit
using Yao, Yao.EasyBuild
using OMEinsum
using OMEinsumContractionOrders
using TensorKit, MPSKit, MPSKitModels, PEPSKit
using MPSKitModels: transverse_field_ising, InfiniteStrip, InfiniteCylinder, 
                    @mpoham, InfiniteChain, nearest_neighbours, vertices
using PEPSKit: InfiniteSquare
using Optimization, OptimizationCMAEvolutionStrategy
using CMAEvolutionStrategy
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
include("gates.jl")
include("training.jl")
include("visualization.jl")
include("reference.jl")
include("exact.jl")

# =============================================================================
# Quantum Channel Simulation
# =============================================================================
export sample_quantum_channel, track_convergence_to_steady_state, estimate_correlation_length_from_sampling, estimate_correlation_length_exact

# =============================================================================
# Gate Construction & Energy
# =============================================================================
export build_unitary_gate, compute_energy

# =============================================================================
# Optimization / Training
# =============================================================================
export CircuitOptimizationResult, ExactOptimizationResult, ManifoldOptimizationResult
export optimize_circuit, optimize_exact, optimize_manifold

# =============================================================================
# Exact Tensor Contraction
# =============================================================================
export compute_transfer_spectrum, compute_single_transfer
export compute_X_expectation, compute_ZZ_expectation
export contract_transfer_matrix, compute_exact_energy, gates_to_tensors
export get_transfer_matrix

# =============================================================================
# Reference Implementations
# =============================================================================
export mpskit_ground_state, mpskit_ground_state_1d, pepskit_ground_state

# =============================================================================
# Data I/O & Visualization
# =============================================================================
export save_result, load_result, save_results, load_results
export plot_acf, compute_acf, fit_acf, fit_acf_exponential
export plot_training_history, plot_variance_vs_samples
export plot_corr_scale
export plot_eigenvalue_spectrum, reconstruct_gates
end