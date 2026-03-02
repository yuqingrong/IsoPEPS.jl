module ITensorsExt

using IsoPEPS
using ITensors
using ITensorInfiniteMPS
using KrylovKit
using LinearAlgebra

# Helper: convert MPS tensor to ITensor format
function _to_ITensor(mps_tensor, row, virtual_qubits)
    physical_dim, left_bond, right_bond = size(mps_tensor)
    s = Index(physical_dim, "Site,n=1")
    l = Index(left_bond, "Link,l=0")
    r = Index(right_bond, "Link,l=1")
    A_permuted = permutedims(mps_tensor, (2, 1, 3))
    A_itensor = ITensor(A_permuted, l, s, r)
    return A_itensor, (site=s, left=l, right=r)
end

function IsoPEPS.transfer_matrix_ITensor(gates, row, virtual_qubits; num_eigenvalues=64)
    mps_tensor = IsoPEPS.reshape_to_mps(gates, row, virtual_qubits)
    A, indices = _to_ITensor(mps_tensor, row, virtual_qubits)

    s, l, r = indices.site, indices.left, indices.right

    l_prime = prime(l)
    r_prime = prime(r)
    A_dag = dag(A)
    A_dag = replaceinds(A_dag, [l, r], [l_prime, r_prime])

    T = A * A_dag

    left_comb = combiner(l, l_prime)
    right_comb = combiner(r, r_prime)
    T_matrix = T * left_comb * right_comb

    T_array = Array(T_matrix, combinedind(left_comb), combinedind(right_comb))

    matrix_size = size(T_array, 1)
    if matrix_size > 256
        vals, _, _ = KrylovKit.eigsolve(T_array, num_eigenvalues, :LM;
                                        ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
    else
        eig_result = LinearAlgebra.eigen(T_array)
        vals = eig_result.values
    end

    sorted_idx = sortperm(abs.(vals), rev=true)
    eigenvalues = vals[sorted_idx][1:min(num_eigenvalues, length(vals))]

    correlation_length = length(eigenvalues) > 1 ? -1.0 / log(abs(eigenvalues[2] / eigenvalues[1])) : Inf

    return T_matrix, eigenvalues, correlation_length
end

end # module ITensorsExt
