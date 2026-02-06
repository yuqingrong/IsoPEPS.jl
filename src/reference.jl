"""
    mpskit_ground_state(d, D, g, row)

Compute exact ground state using MPSKit (VUMPS algorithm).

# Arguments
- `d`: Physical dimension (usually 2)
- `D`: Bond dimension
- `g`: Transverse field strength
- `row`: Number of rows (cylinder circumference)

# Returns
Named tuple with:
- `energy`: Ground state energy per site
- `correlation_length`: Correlation length
- `entropy`: Entanglement entropy
- `spectrum`: Transfer matrix spectrum

# Description
Uses VUMPS from MPSKit to find the ground state of the transverse field 
Ising model on an infinite cylinder: H = -g∑Xᵢ - ∑ZᵢZⱼ
"""
function mpskit_ground_state(d::Int, D::Int, g::Float64, row::Int)
    mps = MPSKit.InfiniteMPS([ComplexSpace(d) for _ in 1:row], [ComplexSpace(D) for _ in 1:row])
    H = transverse_field_ising(InfiniteCylinder(row); g=g)
    psi, _ = find_groundstate(mps, H, VUMPS())
    
    E = real(expectation_value(psi, H)) / row
    spectrum = transfer_spectrum(psi)
    corr_lengths = correlation_length(psi)
    
    len = isempty(corr_lengths) ? NaN : corr_lengths[1]
    entropy = MPSKit.entropy(psi)
    
    return (energy=E, correlation_length=len, entropy=entropy, spectrum=spectrum)
end

"""
    mpskit_ground_state_1d(d, D, g)

Compute 1D chain ground state using MPSKit.

# Arguments
- `d`: Physical dimension
- `D`: Bond dimension
- `g`: Transverse field strength

# Returns
Named tuple with energy, correlation_length, entropy, spectrum, and psi (wavefunction)
"""
function mpskit_ground_state_1d(d::Int, D::Int, g::Float64)
    mps = MPSKit.InfiniteMPS([ComplexSpace(d)], [ComplexSpace(D)])
    H = transverse_field_ising(; g=g)
    psi, _ = find_groundstate(mps, H, VUMPS())
    
    E = real(expectation_value(psi, H))
    spectrum = transfer_spectrum(psi)
    corr_lengths = correlation_length(psi)
    
    len = isempty(corr_lengths) ? NaN : corr_lengths[1]
    entropy = MPSKit.entropy(psi)
    
    return (energy=E, correlation_length=len, entropy=entropy, spectrum=spectrum, psi=psi)
end

"""
    pepskit_ground_state(d, D, J, g; χ=20, ctmrg_tol=1e-10, grad_tol=1e-6, maxiter=1000)

Compute 2D PEPS ground state using PEPSKit.

# Arguments
- `d`: Physical dimension
- `D`: PEPS bond dimension
- `J`: Coupling strength
- `g`: Transverse field strength
- `χ`: Environment bond dimension for CTMRG (default: 20)
- `ctmrg_tol`: CTMRG convergence tolerance (default: 1e-10)
- `grad_tol`: Gradient tolerance (default: 1e-6)
- `maxiter`: Maximum iterations (default: 1000)

# Returns
Named tuple with:
- `energy`: Ground state energy per site
- `correlation_length`: Correlation length (maximum of horizontal and vertical)
- `ξ_horizontal`: Correlation lengths in horizontal direction
- `ξ_vertical`: Correlation lengths in vertical direction
- `peps`: Optimized PEPS state
- `env`: CTMRG environment
"""
function pepskit_ground_state(d::Int, D::Int, J::Float64, g::Float64; 
                               χ::Int=20, ctmrg_tol::Float64=1e-10, 
                               grad_tol::Float64=1e-6, maxiter::Int=1000)
    H = transverse_field_ising(PEPSKit.InfiniteSquare(); g=g)
    peps₀ = InfinitePEPS(ComplexSpace(d), ComplexSpace(D))
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χ)), peps₀; tol=ctmrg_tol)
    
    peps, env, E, = fixedpoint(H, peps₀, env₀; 
                                tol=grad_tol, 
                                boundary_alg=(; tol=ctmrg_tol), 
                                optimizer_alg=(; maxiter=maxiter))

    
    return E, peps
end

"""
    complete_to_unitary(V::Matrix)

Complete an isometry matrix to a unitary by adding orthogonal rows/columns from the nullspace.

# Arguments
- `V`: Isometry matrix (either V*V' = I for wide matrix, or V'*V = I for tall matrix)

# Returns
- Square unitary matrix U containing V as a submatrix

# Description
For a wide matrix (m < n): V*V' = I, completes by adding (n-m) orthogonal rows.
For a tall matrix (m > n): V'*V = I, completes by adding (m-n) orthogonal columns.
"""
function complete_to_unitary(V::Matrix)
    m, n = size(V)
    if m == n
        return V  # Already square
    elseif m < n
        # Wide matrix (m rows, n cols): V*V' = I (rows are orthonormal)
        # Find nullspace of V (vectors orthogonal to all rows)
        V_null = nullspace(V)'  # Transpose to get rows
        return vcat(V, V_null)
    else
        # Tall matrix (m rows, n cols): V'*V = I (columns are orthonormal)
        # Find nullspace of V' (vectors orthogonal to all columns)
        V_null = nullspace(V')
        return hcat(V, V_null)
    end
end

"""
    optimize_peps_gate(; d=2, D=2, J=1.0, g=1.0, save_path="data",
                        χ=20, ctmrg_tol=1e-10, grad_tol=1e-6, maxiter=1000)

Optimize PEPS ground state, extract tensor, complete nullspace to unitary gate, and save.

# Arguments
- `d`: Physical dimension (default: 2)
- `D`: PEPS bond dimension (default: 2)
- `J`: Coupling strength (default: 1.0)
- `g`: Transverse field strength (default: 1.0)
- `save_path`: Directory to save gate file (default: "data")
- `χ`: Environment bond dimension for CTMRG (default: 20)
- `ctmrg_tol`: CTMRG convergence tolerance (default: 1e-10)
- `grad_tol`: Gradient tolerance for optimization (default: 1e-6)
- `maxiter`: Maximum optimization iterations (default: 1000)

# Returns
- `gate_matrix`: The completed unitary gate
- `E`: Ground state energy from PEPS optimization
- `filename`: Path to saved gate file

# Saved File Contents
- Gate matrix (as nested arrays)
- PEPS tensor (original, before completion)
- Energy, parameters (d, D, J, g), nqubits

# Example
```julia
gate, E, filename = optimize_peps_gate(; D=2, J=1.0, g=2.0)
```
"""
function optimize_peps_gate(; d::Int=2, D::Int=2, J::Float64=1.0, g::Float64=1.0,
                             save_path::String="data", χ::Int=20, ctmrg_tol::Float64=1e-10,
                             grad_tol::Float64=1e-6, maxiter::Int=1000)
    
    println("=== Optimize PEPS Gate ===")
    println("Parameters: d=$d, D=$D, J=$J, g=$g")
    
    # Step 1: Get PEPS ground state
    println("\n[1/4] Optimizing PEPS ground state...")
    E, peps = pepskit_ground_state(d, D, J, g; χ=χ, ctmrg_tol=ctmrg_tol, 
                                    grad_tol=grad_tol, maxiter=maxiter)
    println("Ground state energy: $E")
    
    # Step 2: Extract unit tensor
    println("\n[2/4] Extracting unit tensor...")
    A = peps.A[1, 1]  # TensorKit TensorMap
    A_array = Array(A)  # Convert to Julia array
    # Leg ordering: [physical, down, right, up, left] = [d, D, D, D, D]
    println("Tensor shape: $(size(A_array))")
    
    # Step 3: Reshape to isometry matrix
    # Input legs: [physical, down, right] → d × D × D = d*D^2
    # Output legs: [up, left] → D × D = D^2
    println("\n[3/4] Reshaping to isometry matrix...")
    
    # Permute to [up, left, phy, down, right] then reshape
    A_perm = permutedims(A_array, (4, 5, 1, 2, 3))  # [up, left, phy, down, right]
    out_dim = D * D  # up × left
    in_dim = d * D * D  # phy × down × right
    A_matrix = reshape(A_perm, out_dim, in_dim)
    println("Isometry matrix shape: $(size(A_matrix))")
    
    # Verify isometry property: A*A' ≈ I (for wide matrix)
    AAdag = A_matrix * A_matrix'
    isometry_error = norm(AAdag - I(out_dim))
    println("Isometry error ||A*A' - I||: $isometry_error")
    
    # Step 4: Complete nullspace to unitary
    println("\n[4/4] Completing nullspace to unitary...")
    gate_matrix = complete_to_unitary(A_matrix)
    println("Unitary gate shape: $(size(gate_matrix))")
    
    # Verify unitarity
    UUdag = gate_matrix * gate_matrix'
    unitarity_error = norm(UUdag - I(size(gate_matrix, 1)))
    println("Unitarity error ||U*U' - I||: $unitarity_error")
    
    # Determine nqubits from gate size
    gate_dim = size(gate_matrix, 1)
    nqubits = Int(log2(gate_dim))
    println("Gate acts on $nqubits qubits (dimension $gate_dim)")
    
    # Save gate to file
    !isdir(save_path) && mkpath(save_path)
    filename = joinpath(save_path, "peps_gate_J=$(J)_g=$(g)_D=$(D).json")
    
    # Convert gate matrix to nested arrays for JSON serialization
    gate_nested = [collect(gate_matrix[i, :]) for i in 1:size(gate_matrix, 1)]
    # Also save original PEPS tensor
    A_nested = [collect(vec(A_array[:, :, :, :, i])) for i in 1:size(A_array, 5)]
    
    gate_data = Dict{Symbol, Any}(
        :type => "PEPSGate",
        :gate => gate_nested,
        :gate_shape => size(gate_matrix),
        :peps_tensor => A_nested,
        :peps_tensor_shape => size(A_array),
        :peps_energy => E,
        :isometry_error => isometry_error,
        :unitarity_error => unitarity_error,
        :params => Dict{Symbol, Any}(
            :d => d,
            :D => D,
            :J => J,
            :g => g,
            :nqubits => nqubits,
            :χ => χ,
            :ctmrg_tol => ctmrg_tol,
            :grad_tol => grad_tol,
            :maxiter => maxiter
        )
    )
    
    open(filename, "w") do io
        JSON3.pretty(io, gate_data)
    end
    println("\nGate saved to: $filename")
    
    return gate_matrix, E, filename
end

"""
    load_peps_gate(filename::String)

Load a saved PEPS gate from JSON file.

# Arguments
- `filename`: Path to the saved gate file

# Returns
- `gate_matrix`: The unitary gate matrix
- `params`: Dictionary of parameters (d, D, J, g, nqubits, etc.)
- `E`: PEPS ground state energy
"""
function load_peps_gate(filename::String)
    data = open(filename, "r") do io
        JSON3.read(io, Dict)
    end
    
    # Reconstruct gate matrix from nested arrays
    gate_nested = data["gate"]
    gate_shape = Tuple(data["gate_shape"])
    gate_matrix = Matrix{ComplexF64}(undef, gate_shape...)
    for i in 1:gate_shape[1]
        gate_matrix[i, :] = ComplexF64.(gate_nested[i])
    end
    
    # Extract parameters
    params_raw = data["params"]
    params = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(params_raw))
    
    E = data["peps_energy"]
    
    return gate_matrix, params, E
end

"""
    sample_peps_gate(filename::String; row=1, conv_step=1000, samples=10000, 
                     measure_first=:Z, save_results=true, save_path="data")

Load a saved PEPS gate and run quantum channel sampling.

# Arguments
- `filename`: Path to saved gate file (from `optimize_peps_gate`)
- `row`: Number of rows for the quantum channel (default: 1)
- `conv_step`: Convergence steps before sampling (default: 1000)
- `samples`: Number of samples to collect (default: 10000)
- `measure_first`: Which observable to measure first, :X or :Z (default: :Z)
- `save_results`: Whether to save sampling results (default: true)
- `save_path`: Directory to save results (default: "data")

# Returns
- `rho`: Final quantum state
- `Z_samples`: Vector of Z measurement outcomes
- `X_samples`: Vector of X measurement outcomes
- `energy`: Sampled energy

# Example
```julia
# First optimize and save gate
gate, E, gate_file = optimize_peps_gate(; D=2, J=1.0, g=2.0)

# Then sample (can run multiple times with different settings)
rho, Z, X, energy = sample_peps_gate(gate_file; samples=50000)
rho, Z, X, energy = sample_peps_gate(gate_file; samples=100000, conv_step=5000)
```
"""
function sample_peps_gate(filename::String; row::Int=1, conv_step::Int=1000, 
                          samples::Int=10000, measure_first::Symbol=:Z,
                          save_results::Bool=true, save_path::String="data")
    
    println("=== Sample PEPS Gate ===")
    println("Loading gate from: $filename")
    
    # Load gate
    gate_matrix, params, E = load_peps_gate(filename)
    
    d = params[:d]
    D = params[:D]
    J = params[:J]
    g = params[:g]
    nqubits = params[:nqubits]
    
    println("Parameters: d=$d, D=$D, J=$J, g=$g, nqubits=$nqubits")
    println("PEPS energy: $E")
    println("Sampling: row=$row, conv_step=$conv_step, samples=$samples")
    
    # Create gate vector (same gate for all rows)
    gates = [Matrix{ComplexF64}(gate_matrix) for _ in 1:row]
    
    # Run quantum channel
    println("\nRunning quantum channel sampling...")
    rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits;
                                                        conv_step=conv_step,
                                                        samples=samples,
                                                        measure_first=measure_first)
    
    println("Generated $(length(Z_samples)) Z samples and $(length(X_samples)) X samples")
    
    # Compute energy from samples
    energy = compute_energy(X_samples, Z_samples, g, J, row)
    println("Sampled energy: $energy")
    println("PEPS energy: $E")
    println("Energy difference: $(abs(energy - E))")
    
    # Save results if requested
    if save_results
        !isdir(save_path) && mkpath(save_path)
        result_filename = joinpath(save_path, "peps_samples_J=$(J)_g=$(g)_D=$(D)_row=$(row).json")
        
        Z_nested = [Z_samples]
        X_nested = [X_samples]
        
        result_data = Dict{Symbol, Any}(
            :type => "PEPSSamplingResult",
            :gate_file => filename,
            :energy => energy,
            :peps_energy => E,
            :Z_samples => Z_nested,
            :X_samples => X_nested,
            :sample_shape => (1, length(Z_samples)),
            :input_args => Dict{Symbol, Any}(
                :d => d,
                :D => D,
                :J => J,
                :g => g,
                :row => row,
                :nqubits => nqubits,
                :conv_step => conv_step,
                :samples => samples,
                :measure_first => String(measure_first),
                :source => "sample_peps_gate"
            )
        )
        
        open(result_filename, "w") do io
            JSON3.pretty(io, result_data)
        end
        println("\nResults saved to: $result_filename")
    end
    
    return rho, Z_samples, X_samples, energy
end