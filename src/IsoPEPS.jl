"""
    IsoPEPS

A Julia package for Isometric Projected Entangled Pair States (IsoPEPS) algorithms.

# Features
- Quantum channel simulation for PEPS
- Transfer matrix calculations with exact contraction
- Variational optimization (CMA-ES, manifold methods)
- Reference implementations using MPSKit and PEPSKit (via extensions)
- Data visualization and analysis (via CairoMakie extension)

# Main Functions
- `sample_quantum_channel`: Sample observables from quantum channel
- `build_unitary_gate`: Build parameterized unitary gates
- `compute_energy`: Compute TFIM energy from samples
- `optimize_circuit`: Optimize circuit parameters
- `compute_transfer_spectrum`: Compute transfer matrix spectrum
- `mpskit_ground_state`: Reference ground state from MPSKit (requires `using MPSKit, TensorKit, MPSKitModels`)

# Extensions
Load additional functionality by importing the trigger packages:
- `using CairoMakie` → plotting functions (plot_acf, plot_training_history, etc.)
- `using LsqFit` → ACF fitting (fit_acf, fit_acf_oscillatory)
- `using Manifolds, Manopt` → manifold optimization (optimize_manifold)
- `using TensorKit, MPSKit, MPSKitModels` → MPSKit reference (mpskit_ground_state)
- `using PEPSKit, TensorKit, MPSKitModels` → PEPSKit reference (pepskit_ground_state)
- `using ITensors` → ITensor transfer matrix (transfer_matrix_ITensor)
"""
module IsoPEPS

# =============================================================================
# Core Dependencies
# =============================================================================
using LinearAlgebra
using Statistics
using KrylovKit
using Yao, Yao.EasyBuild
using OMEinsum
using OMEinsumContractionOrders
using Optimization
using OptimizationCMAEvolutionStrategy
using CMAEvolutionStrategy
using Optim
using JSON3
using CairoMakie
using LsqFit

# =============================================================================
# Index helper for tensor contractions
# =============================================================================
mutable struct IndexStore
    count::Int
    IndexStore() = new(0)
end

function newindex!(store::IndexStore)
    store.count += 1
    return store.count
end

# =============================================================================
# Include core module files
# =============================================================================
include("quantum_channels.jl")
include("gates.jl")
include("training.jl")
include("visualization.jl")
include("transfer_matrix.jl")
include("observables.jl")

# =============================================================================
# Function stubs for extensions (methods added by package extensions)
# =============================================================================

# --- CairoMakie extension ---

# --- LsqFit extension ---

# --- Manifolds + Manopt extension ---
function optimize_manifold end

# --- TensorKit + MPSKit + MPSKitModels extension ---
function mpskit_ground_state end
function mpskit_ground_state_1d end
function spectrum_MPSKit end

# --- PEPSKit extension ---
function pepskit_ground_state end

# --- ITensors extension ---
function transfer_matrix_ITensor end

# =============================================================================
# Exports
# =============================================================================

# Quantum Channel Simulation
export sample_quantum_channel, track_convergence_to_steady_state, estimate_correlation_length_from_sampling, estimate_correlation_length_exact

# Gate Construction & Energy
export build_unitary_gate, build_unitary_gate_2x2, compute_energy, compute_heisenberg_energy

# Optimization / Training
export CircuitOptimizationResult, ExactOptimizationResult, ManifoldOptimizationResult
export optimize_circuit, optimize_exact, optimize_manifold, initialize_tfim_params

# Exact Tensor Contraction - Transfer Matrix
export compute_transfer_spectrum, compute_single_transfer
export get_combined_transfer_matrix, compute_transfer_spectrum_2x2
export contract_transfer_matrix, gates_to_tensors, get_transfer_matrix, get_physical_channel
export build_transfer_code, build_physical_channel_code, apply_transfer_matvec
export get_transfer_matrix_with_operator, compute_correlation_coefficients
export compute_theoretical_correlation_decay, compute_theoretical_lambda_eff
export reshape_to_mps, spectrum_MPSKit, transfer_matrix_ITensor

# Exact Tensor Contraction - Observables
export compute_X_expectation, compute_Z_expectation, compute_ZZ_expectation, compute_single_expectation
export compute_exact_energy, compute_exact_heisenberg_energy, compute_exact_heisenberg_energy_2x2, intercolumn_correlation
export correlation_function, expect

# Reference Implementations (loaded via extensions)
export mpskit_ground_state, mpskit_ground_state_1d, pepskit_ground_state

# Data I/O
export save_result, load_result, save_results, load_results, resample_circuit
export reconstruct_gates, compute_acf

# Visualization (loaded via CairoMakie extension)
export plot_acf, fit_acf, fit_acf_oscillatory
export plot_training_history, plot_variance_vs_samples, plot_expectation_values
export plot_corr_scale
export plot_eigenvalue_spectrum, plot_correlation_function
export plot_energy_error_vs_g,plot_correlation_vs_g
end
