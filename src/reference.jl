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
- `psi`: Optimized InfiniteMPS wavefunction

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
    
    return (energy=E, correlation_length=len, entropy=entropy, spectrum=spectrum, psi=psi)
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
    A = peps.A[1, 1]  # TensorKit TensorMap: P ← N ⊗ E ⊗ S ⊗ W
    A_array = convert(Array, A)  # Convert to Julia array
    # PEPSKit leg ordering: [physical, N, E, S, W] = [physical, up, right, down, left]
    println("Tensor shape (P,N,E,S,W): $(size(A_array))")
    
    # Step 3: Reshape PEPS tensor to isometric form via QR decomposition
    # IsoPEPS convention: tensor legs [physical, down, right, up, left]
    # Gate tensor: G[p_out, down, right, p_in, up, left]
    # Selecting p_in=1 (|0⟩): G[:,:,:,1,:,:] = A[physical, down, right, up, left]
    # This is an isometry from (up, left) → (physical, down, right)
    println("\n[3/4] Building isometric form via QR...")
    
    # Permute from PEPSKit [P, N, E, S, W] to IsoPEPS [physical, down, right, up, left]
    A_perm = permutedims(A_array, (1,4,5,2,3))  # [physical, down, right, up, left]
    out_dim = d * D * D   # physical × down × right (gate output legs)
    in_dim = D * D        # up × left (gate input legs)
    A_matrix = reshape(A_perm, out_dim, in_dim)  # tall matrix: (d*D², D²)
    println("Tensor reshaped to matrix: $(size(A_matrix))")
    
    # PEPSKit gives a general (non-isometric) PEPS tensor.
    # QR decomposition: A = Q*R, where Q is an isometry (Q'Q = I).
    # Q has the same column space as A, so it encodes the same physical state
    # up to a gauge transformation R on the virtual legs (up, left).
    # This is exact — no approximation is made.
    F_qr = qr(A_matrix)
    A_iso = Matrix(F_qr.Q)  # isometry: (d*D² × D²), A_iso'*A_iso = I
    R_gauge = Matrix(F_qr.R)  # gauge: (D² × D²)
    
    raw_error = norm(A_matrix' * A_matrix - I(in_dim))
    iso_error = norm(A_iso' * A_iso - I(in_dim))
    println("Raw tensor isometry error ||A'A - I||: $(raw_error)")
    println("After QR: ||Q'Q - I||: $(iso_error)")
    println("Gauge matrix R condition number: $(cond(R_gauge))")
    println("Reconstruction error ||A - Q*R||: $(norm(A_matrix - A_iso * R_gauge))")
    
    # Step 4: Build full unitary gate
    # Place isometric tensor at p_in=1 slice, fill remaining with orthogonal complement
    println("\n[4/4] Completing to unitary gate...")
    gate_dim = d * D * D  # gate dimension = d*D²
    
    # Build 6-index gate tensor and place isometry at p_in=1
    gate_tensor = zeros(ComplexF64, d, D, D, d, D, D)
    gate_tensor[:, :, :, 1, :, :] = reshape(A_iso, d, D, D, D, D)
    gate_matrix = reshape(gate_tensor, gate_dim, gate_dim)
    
    # Find orthogonal complement for the zero columns (p_in > 1)
    V_null = nullspace(gate_matrix')
    zero_cols = [j for j in 1:gate_dim if all(gate_matrix[:, j] .== 0)]
    for (i, col) in enumerate(zero_cols)
        gate_matrix[:, col] = V_null[:, i]
    end
    println("Unitary gate shape: $(size(gate_matrix))")
    
    # Verify unitarity
    UUdag = gate_matrix * gate_matrix'
    unitarity_error = norm(UUdag - I(gate_dim))
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
        :raw_isometry_error => raw_error,
        :polar_isometry_error => iso_error,
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
    optimize_mps_gate(; d=2, D=2, J=1.0, g=1.0, row=1, save_path="data")

Optimize MPS ground state on an infinite cylinder, reshape the MPS tensor into a
PEPS-like 5-index tensor, complete to a unitary gate, and save.

Instead of PEPSKit (which gives a non-isometric tensor), this uses MPSKit (VUMPS)
on a cylinder. The MPS virtual bond dimension is D_mps = D^2, so that the virtual
legs can be split into (up, left) and (down, right) pairs of dimension D each,
recovering a PEPS-like tensor.

The MPS tensor from VUMPS (right-canonical AR form) is already an isometry, so no
QR approximation step is needed.

# Arguments
- `d`: Physical dimension (default: 2)
- `D`: PEPS bond dimension (default: 2). MPS bond dimension will be D^2.
- `J`: Coupling strength (default: 1.0)
- `g`: Transverse field strength (default: 1.0)
- `row`: Number of rows / cylinder circumference (default: 1)
- `save_path`: Directory to save gate file (default: "data")

# Returns
- `gate_matrix`: The completed unitary gate
- `E`: Ground state energy per site
- `filename`: Path to saved gate file

# Example
```julia
gate, E, filename = optimize_mps_gate(; D=2, J=1.0, g=2.0, row=1)
```
"""
function optimize_mps_gate(; d::Int=2, D::Int=2, J::Float64=1.0, g::Float64=1.0,
                             row::Int=1, save_path::String="data")
    
    D_mps = D^2  # MPS bond dim = D_peps^2
    println("=== Optimize MPS Gate ===")
    println("Parameters: d=$d, D_peps=$D, D_mps=$D_mps, J=$J, g=$g, row=$row")

    println("\n[1/4] Optimizing MPS ground state on cylinder (row=$row)...")
    result = mpskit_ground_state(d, D_mps, g, row)
    E = result.energy
    psi = result.psi
    
    println("\n[2/4] Extracting MPS tensor (AR form)...")
    
    A_map = convert(Array, psi.AR[1])
    A_map = permutedims(A_map, (2, 3, 1))

    A_check = reshape(A_map, d*D_mps, D_mps)
    @assert isapprox(A_check' * A_check, I(D_mps), atol=1e-5)
   
        
    nullspace_A = LinearAlgebra.nullspace(A_check')
    A_matrix = hcat(A_check, nullspace_A) 
    @show size(A_matrix)
    A_mod = similar(A_matrix)
    A_mod[:, 1] = A_matrix[:, 1]
    A_mod[:, 3] = A_matrix[:, 2]
    A_mod[:, 5] = A_matrix[:, 3]
    A_mod[:, 7] = A_matrix[:, 4]
    A_mod[:, 2] = A_matrix[:, 5]
    A_mod[:, 4] = A_matrix[:, 6]
    A_mod[:, 6] = A_matrix[:, 7]
    A_mod[:, 8] = A_matrix[:, 8]
    
    # Determine nqubits from gate size
    gate_dim = size(A_mod, 1)
    nqubits = Int(log2(gate_dim))
    println("Gate acts on $nqubits qubits (dimension $gate_dim)")
    
    # Save gate to file (same format as optimize_peps_gate for compatibility)
    !isdir(save_path) && mkpath(save_path)
    filename = joinpath(save_path, "mps_gate_J=$(J)_g=$(g)_D=$(D)_row=$(row).json")
    gate_matrix = A_mod
    # Convert gate matrix to nested arrays for JSON serialization
    gate_nested = [collect(gate_matrix[i, :]) for i in 1:size(gate_matrix, 1)]
    # Also save original MPS tensor - A_map is 3D: (left, phys, right)
    # Serialize along right index to match PEPS format convention
    A_nested = [collect(vec(A_map[:, :, i])) for i in 1:size(A_map, 3)]
    
    
    gate_tensors = gates_to_tensors([gate_matrix], 1, 1) 
    extracted_tensor = gate_tensors[1]  

    @assert reshape(extracted_tensor, size(A_map)) == A_map
   
    gate_data = Dict{Symbol, Any}(
        :type => "MPSGate",  
        :gate => gate_nested,
        :gate_shape => size(gate_matrix),
        :mps_tensor => A_nested,
        :mps_tensor_shape => size(A_map),
        :mps_energy => E,
        :params => Dict{Symbol, Any}(
            :d => d,
            :D => D,
            :D_mps => D_mps,
            :J => J,
            :g => g,
            :nqubits => nqubits,
            :row => row
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

Load a saved PEPS or MPS gate from JSON file.

# Arguments
- `filename`: Path to the saved gate file

# Returns
- `gate_matrix`: The unitary gate matrix
- `params`: Dictionary of parameters (d, D, J, g, nqubits, etc.)
- `E`: Ground state energy (PEPS or MPS)
"""
function load_peps_gate(filename::String)
    data = open(filename, "r") do io
        JSON3.read(io, Dict)
    end
    
    # Reconstruct gate matrix from nested arrays
    # JSON3 serializes ComplexF64 as Dict with "re"/"im" keys
    _to_complex(x::Number) = ComplexF64(x)
    _to_complex(x) = ComplexF64(x["re"], x["im"])
    
    gate_nested = data["gate"]
    gate_shape = Tuple(data["gate_shape"])
    gate_matrix = Matrix{ComplexF64}(undef, gate_shape...)
    for i in 1:gate_shape[1]
        gate_matrix[i, :] = [_to_complex(v) for v in gate_nested[i]]
    end
    
    # Extract parameters
    params_raw = data["params"]
    params = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(params_raw))
    
    # Handle both PEPS and MPS gate files
    gate_type = get(data, "type", "PEPSGate")
    E = if gate_type == "MPSGate"
        data["mps_energy"]
    else
        data["peps_energy"]
    end
    
    return gate_matrix, params, E
end

"""
    sample_peps_gate(filename::String; row=1, conv_step=1000, samples=10000, 
                     measure_first=:Z, save_results=true, save_path="data")

Load a saved PEPS or MPS gate and run quantum channel sampling.

# Arguments
- `filename`: Path to saved gate file (from `optimize_peps_gate` or `optimize_mps_gate`)
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
- `energy_sampled`: Energy computed from measurement samples
- `energy_exact`: Exact energy computed from gate via tensor contraction

# Example
```julia
# First optimize and save gate
gate, E, gate_file = optimize_peps_gate(; D=2, J=1.0, g=2.0)

# Then sample (can run multiple times with different settings)
rho, Z, X, E_sampled, E_exact = sample_peps_gate(gate_file; samples=50000)
println("Exact energy from gate: ", E_exact)
println("Sampled energy: ", E_sampled)
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
    energy_sampled = compute_energy(X_samples, Z_samples, g, J, row)
    
    # Compute exact energy from gate using tensor contraction
    println("\nComputing exact energy from gate (tensor contraction)...")
    virtual_qubits = (nqubits - 1) ÷ 2
    rho_exact, gap_exact, eigenvalues_exact = compute_transfer_spectrum(gates, row, nqubits)
    
    X_exact = real(compute_X_expectation(rho_exact, gates, row, virtual_qubits))
    ZZ_vert_exact, ZZ_horiz_exact = compute_ZZ_expectation(rho_exact, gates, row, virtual_qubits)
    ZZ_vert_exact = real(ZZ_vert_exact)
    ZZ_horiz_exact = real(ZZ_horiz_exact)
    
    energy_exact = -g * X_exact - J * (row == 1 ? ZZ_horiz_exact : ZZ_vert_exact + ZZ_horiz_exact)
    
    println("\n=== Energy Comparison ===")
    println("Reference energy (PEPS/MPS):  $E")
    println("Exact energy (from gate):     $energy_exact")
    println("Sampled energy:                $energy_sampled")
    println("Gate vs Reference diff:        $(abs(energy_exact - E))")
    println("Sampled vs Exact diff:         $(abs(energy_sampled - energy_exact))")
    println("Spectral gap:                  $gap_exact")
    
    energy = energy_sampled  # Keep sampled for backward compatibility
    
    # Save results if requested
    if save_results
        !isdir(save_path) && mkpath(save_path)
        result_filename = joinpath(save_path, "peps_samples_J=$(J)_g=$(g)_D=$(D)_row=$(row).json")
        
        Z_nested = [Z_samples]
        X_nested = [X_samples]
        
        result_data = Dict{Symbol, Any}(
            :type => "PEPSSamplingResult",
            :gate_file => filename,
            :energy_sampled => energy_sampled,
            :energy_exact => energy_exact,
            :reference_energy => E,
            :spectral_gap => gap_exact,
            :Z_samples => Z_nested,
            :X_samples => X_nested,
            :sample_shape => (1, length(Z_samples)),
            :observables_exact => Dict{Symbol, Any}(
                :X => X_exact,
                :ZZ_vert => ZZ_vert_exact,
                :ZZ_horiz => ZZ_horiz_exact
            ),
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
    
    return rho, Z_samples, X_samples, energy_sampled, energy_exact
end