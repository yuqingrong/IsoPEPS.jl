mutable struct MPO{T, AT<:AbstractArray{T,4}} 
    tensors::Vector{AT}
    canonical_center::Int  # if canonical_center == 0, the MPO is not canonicalized
    function MPO(tensors::Vector{AT}, canonical_center::Int=0) where {T, AT<:AbstractArray{T,4}}
        n = length(tensors)
        nflavor = size(tensors[1], 2)
        @assert n > 0
        @assert canonical_center > 0 && canonical_center <= n || canonical_center == 0
        @assert all(i->size(tensors[i], 2) == size(tensors[i], 3) == nflavor && size(tensors[i], 4) == size(tensors[mod1(i+1, n)], 1), 1:n)
        new{T, AT}(tensors, canonical_center)
    end
end

"""
function generate_mpo(::Type{T}, bondims::Vector{Int}; d::Int=2) where T
    tensors = [make_hermitian(randn(T, 1, d, d, bondims[1]))]
    for i = 2:length(bondims)
        push!(tensors, make_hermitian(randn(T, bondims[i-1], d, d, bondims[i])))
    end
    push!(tensors, make_hermitian(randn(T, bondims[end], d, d, 1)))
    return MPO(tensors)
end
make_hermitian(t::AbstractArray{T, 4}) where T = (t .+= permutedims(conj.(t), (1, 3, 2, 4)); t)
generate_mpo(bondims::Int; d::Int=2) = generate_mpo_mpo(FloatF64, bondims; d)
"""




function transverse_ising_mpo(::Type{T}, n::Int, h::Float64) where T
    @assert n > 1
    tensor1 = zeros(T, 1, 2, 2, 3)
    tensor2 = zeros(T, 3, 2, 2, 3)   
    tensor3 = zeros(T, 3, 2, 2, 1)
    tensor1[1, :, :, 1] = -Matrix{T}(I2)
    tensor2[1, :, :, 1] = tensor2[3, :, :, 3] = tensor3[3, :, :, 1] = Matrix{T}(I2)
    tensor1[1, :, :, 2] = -Matrix{T}(Z)
    tensor2[2, :, :, 3] = tensor2[1, :, :, 2] = tensor3[2, :, :, 1] = Matrix{T}(Z)
    tensor1[1, :, :, 3] = -Matrix{T}(X)* h 
    tensor2[1, :, :, 3] = tensor3[1, :, :, 1] = Matrix{T}(X)* h
    MPO([tensor1, fill(tensor2, n-2)..., tensor3])
end

transverse_ising_mpo(n::Int,h::Float64)=transverse_ising_mpo(Float64,n,h)






function mat2mpo(m::AbstractMatrix; d=2, Dmax=typemax(Int), atol=1e-10)
    nsite = round(Int, log2(length(m)) ÷ log2(d) ÷ 2)
    @assert d^(2nsite) == length(m) "Matrix length is not a power of the physical dimension ($d), got: $(size(m))"
    n = d^nsite
    state = reshape(m, 1, n, 1, n)
    tensors = typeof(state)[]
    for _ = 1:nsite-1
        n ÷= d
        state = permutedims(reshape(state, (size(state, 1) * d, n, d, n)), (1, 3, 2, 4))
        u, s, v, err = truncated_svd(reshape(state, :, n^2), Dmax, atol)
        push!(tensors, reshape(u, size(u, 1) ÷ (d^2), d, d, size(u, 2)))
        state = reshape(s .* v, size(v, 1), n, 1, n)
    end
    push!(tensors, reshape(state, size(state, 1), d, d, 1))
    return MPO(tensors)
end

function local_X(::Type{T}, nsites::Int; d::Int=2) where T
    tensors = []
    X = [0.0 1.0; 1.0 0.0]
    X_reshaped=reshape(X,1,2,2,1)
    @show typeof(X)
    tensors = [X_reshaped for _ in 1:nsites] 
    @show typeof(tensors)
    return MPO(tensors)
end

local_X(nsites::Int;d::Int=2)=local_X(Float64,nsites;d)
