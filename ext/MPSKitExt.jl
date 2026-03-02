module MPSKitExt

using IsoPEPS
using TensorKit, MPSKit, MPSKitModels
using MPSKitModels: transverse_field_ising, InfiniteStrip, InfiniteCylinder,
                    @mpoham, InfiniteChain, nearest_neighbours, vertices
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

end # module MPSKitExt
