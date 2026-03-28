module MPSKitExt

using IsoPEPS
using TensorKit, MPSKit, MPSKitModels
using MPSKitModels: transverse_field_ising, InfiniteStrip, InfiniteCylinder,
                    @mpoham, InfiniteChain, nearest_neighbours, next_nearest_neighbours,
                    vertices, S_exchange
using KrylovKit
using LinearAlgebra

# Helper: convert MPS tensor to TensorKit TensorMap format
function _to_MPSKit(mps_tensor, row, virtual_qubits)
    physical_dim, left_bond, right_bond = size(mps_tensor)
    A_permuted = permutedims(mps_tensor, (2, 1, 3))
    A_tensormap = TensorMap(A_permuted, ComplexSpace(left_bond),
                            ComplexSpace(physical_dim) ⊗ ComplexSpace(right_bond))
    return A_tensormap
end

function IsoPEPS.mpskit_ground_state(d::Int, D::Int, g::Float64, row::Int)
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

function IsoPEPS.mpskit_ground_state_1d(d::Int, D::Int, g::Float64)
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

function IsoPEPS.spectrum_MPSKit(gates, row, virtual_qubits; num_eigenvalues=64)
    mps_tensor = IsoPEPS.reshape_to_mps(gates, row, virtual_qubits)
    A = _to_MPSKit(mps_tensor, row, virtual_qubits)

    @tensor T[-1 -2; -3 -4] := A[-1 1 -3] * conj(A[-2 1 -4])

    left_dim = TensorKit.dim(codomain(T))
    right_dim = TensorKit.dim(domain(T))
    T_array = reshape(convert(Array, T), left_dim, right_dim)

    if left_dim > 256
        vals, _, _ = KrylovKit.eigsolve(T_array, num_eigenvalues, :LM;
                                        ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
    else
        eig_result = LinearAlgebra.eigen(T_array)
        vals = eig_result.values
    end

    sorted_idx = sortperm(abs.(vals), rev=true)
    spectrum = vals[sorted_idx][1:min(num_eigenvalues, length(vals))]
    corr_length = length(spectrum) > 1 ? -1.0 / log(abs(spectrum[2] / spectrum[1])) : Inf

    return spectrum, corr_length
end

"""
    mpskit_ground_state_j1j2(d, D, J1, J2, row; unit_cell_cols=2, alg=VUMPS())

Compute the ground state of the Heisenberg J1-J2 model on an InfiniteCylinder
with circumference `row` and `unit_cell_cols` columns per MPS unit cell.

H = J1 Σ_{⟨i,j⟩} Sᵢ·Sⱼ + J2 Σ_{⟨⟨i,j⟩⟩} Sᵢ·Sⱼ
"""
function IsoPEPS.mpskit_ground_state_j1j2(
    d::Int, D::Int, J1::Float64, J2::Float64, row::Int;
    unit_cell_cols::Int=2,
    alg=VUMPS(; maxiter=200, tol=1e-8) 
    #alg=IDMRG2(; trscheme=truncdim(D), maxiter=200, tol=1e-8)
)
    N_sites = row * unit_cell_cols
    lattice = InfiniteCylinder(row, N_sites)

    # Build Hamiltonian: H = J1 Σ_nn S·S + J2 Σ_nnn S·S
    nn_op = rmul!(S_exchange(ComplexF64, Trivial; spin=1//2), J1)

    if J2 != 0.0
        nnn_op = rmul!(S_exchange(ComplexF64, Trivial; spin=1//2), J2)
        H = @mpoham begin
            sum(nearest_neighbours(lattice)) do (i, j)
                return nn_op{i, j}
            end + sum(next_nearest_neighbours(lattice)) do (i, j)
                return nnn_op{i, j}
            end
        end
    else
        H = @mpoham sum(nearest_neighbours(lattice)) do (i, j)
            return nn_op{i, j}
        end
    end

    # Initialize and find ground state
    mps = MPSKit.InfiniteMPS(
        [ComplexSpace(d) for _ in 1:N_sites],
        [ComplexSpace(D) for _ in 1:N_sites]
    )

    psi, envs = find_groundstate(mps, H, alg)

    # Compute observables
    E = real(expectation_value(psi, H)) / N_sites
    spectrum = transfer_spectrum(psi)
    corr_lengths = correlation_length(psi)
    len = isempty(corr_lengths) ? NaN : corr_lengths[1]
    entropy = MPSKit.entropy(psi)

    return (energy=E, correlation_length=len, entropy=entropy,
            spectrum=spectrum, psi=psi, H=H)
end

end # module MPSKitExt
