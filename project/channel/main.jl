using Yao, LinearAlgebra
using TensorInference, TensorInference.OMEinsum


function iterate_channel(gate, niters)
    rho1 = density_matrix(zero_state(1))  # the state qubit
    for i=1:niters
        rho2 = join(rho1, density_matrix(zero_state(1)))
        @assert all(iszero, measure(rho2, 1; nshots=1000))
        rho2 = apply!(rho2, gate)
        rho1 = partial_tr(rho2, 1) |> normalize!
        @info "eigenvalues of ρ = " eigen(Hermitian(rho1.state)).values
    end
    return rho1
end

# The tensor inference approach
function train_uai(T::Array{ET}, n::Int) where ET
    nvars = n
    ixs = [[i, mod1(i+1, nvars)] for i=1:nvars]
    tensors = [T for _=1:nvars]
    return UAIModel(
        nvars,
        fill(2, nvars),
        [TensorInference.Factor((ixs[i]...,), tensors[i]) for i in 1:nvars]
    )
end

function circle_mps_bp(gate, n)
    # transfer matrix
    A = reshape(mat(gate), 2, 2, 2, 2)[:, :, 1, :]
    T = reshape(ein"iab,icd->cabd"(conj(A), A), 4, 4)

    mps_uai = train_uai(T, n)
    bp = BeliefPropgation(mps_uai)
    state, info = belief_propagate(bp; max_iter=1000, tol=1e-8)
    @assert info.converged
    st = reshape(state.message_in[1][1], 2, 2)
    @assert st ≈ st' atol=1e-6
    return st
end


function compare(gate)
    # gate = ConstGate.SWAP
    rho = iterate_channel(gate, 100)

    # measure the state qubit
    evals = eigvals(Hermitian(rho.state))
    r1 = evals[1] / evals[2]

    st = circle_mps_bp(gate, 12)
    evals = eigvals(Hermitian(st))
    r2 = evals[1] / evals[2]
    return r1, r2
end

gate = matblock(rand_unitary(ComplexF64, 4))
compare(gate)