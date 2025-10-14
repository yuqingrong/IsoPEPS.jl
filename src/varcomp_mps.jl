""" 1D Ising model analytical result """
function int(h::Float64, J::Float64)
    f(u,p) = sqrt((J-h)^2 + 4*J*h*sin(u/2)^2)/ (-2*π)
    domain= (-π,π)
    prob = IntegralProblem(f,domain)
    sol = solve(prob, HCubatureJL(); abstol=1e-10)
    return sol
end


""" Result from MPSKit.jl """
function exact_energy(d,D,J,g)
    psi0 = InfiniteMPS([ℂ^d], [ℂ^D])
    H0 = transverse_field_ising(;J=J, g=g)
    psi,_= find_groundstate(psi0, H0, VUMPS())
    E = real(expectation_value(psi,H0))
    return psi, E
end


"""
SU(4) gate: U(θ) = exp(-i ∑_{k=1}^{15} θ_k G_k); G = {I,X,Y,Z}⊗{I,X,Y,Z} \\ {I⊗I}  15 parameters
"""
function pauli_generators_2qubit(; parallel=true)
    I_pauli = Matrix(I2)
    X_pauli = Matrix(X)
    Y_pauli = Matrix(Y)
    Z_pauli = Matrix(Z)
    
    single_paulis = [I_pauli, X_pauli, Y_pauli, Z_pauli]
    index_pairs = [(i, j) for i in 1:4, j in 1:4 if !(i == 1 && j == 1)]
    
    if parallel
        generators = Vector{Matrix{ComplexF64}}(undef, length(index_pairs))
        Threads.@threads for i in eachindex(index_pairs)
            pair = index_pairs[i]
            generators[i] = kron(single_paulis[pair[1]], single_paulis[pair[2]])
        end
        return generators
    else
        generators = map(pair -> kron(single_paulis[pair[1]], single_paulis[pair[2]]), index_pairs)
        return generators
    end
end

function su4_gate(θ::Vector{Float64}; parallel=true)
    @assert length(θ) == 15
    generators = pauli_generators_2qubit(parallel=parallel)
    
    if parallel
        H = zeros(ComplexF64, 4, 4)
        partial_sums = Vector{Matrix{ComplexF64}}(undef, 15)
        
        Threads.@threads for k in 1:15
            partial_sums[k] = θ[k] * generators[k]
        end
        
        H = sum(partial_sums)
    else
        H = sum(θ[k] * generators[k] for k in 1:15)
    end
    
    return exp(-im * H)
end


function cost(θ::Vector{Float64}, target_mps::InfiniteMPS, circuit_layer::Int)
    fai = init_prodstate() 
    Vθ_mps = apply_circuit(fai, θ, circuit_layer)   # TODO: complete them

    return 1-sqrt(local_fidelity(target_mps, Vθ_mps))
end

function gradient(θ::Vector{Float64}, target_mps::InfiniteMPS, circuit_layer::Int; ϵ=1e-7)
    grad = zeros(length(θ))
    current_cost = cost(θ, target_mps, circuit_layer)   # Finite difference package
    for i in eachindex(θ)
        θ_plus = copy(θ)
        θ_plus[i] += ϵ
        θ_minus = copy(θ)
        θ_minus[i] -= ϵ
    end
    return grad
end

function local_fidelity(mps1::InfiniteMPS, mps2::InfiniteMPS)
    T = transfer_matrix_2sites(mps1,mps2)
    lambda0 = LinearAlgebra.eigvals(T).values[end]
    return lambda0
end

function transfer_matrix_2sites(mps1::InfiniteMPS, mps2::InfiniteMPS)
    A = mps1..
    B = mps1..
    C = mps2..
    D = mps2..
    ein"iab, ibc, ide, ief -> acdf"(conj(A),conj(B),C,D)
end