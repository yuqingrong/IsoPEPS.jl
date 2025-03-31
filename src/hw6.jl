using LinearAlgebra

function restarting_lanczos(A, q1, s; max_restarts=5, tol=1e-10)
    n = size(A, 1)
    θ1_prev = 0.0
    q1 = normalize(q1)

    for _ in 1:max_restarts
        Q = zeros(n, s)
        α = zeros(s)
        β = zeros(s-1)
        Q[:, 1] = q1

        for j in 1:s-1
            v = A * Q[:, j]
            α[j] = real(dot(Q[:, j], v))
            v = v .- α[j] .* Q[:, j] .- (j > 1 ? β[j-1] .* Q[:, j-1] : 0.0)  # Fixed line
            β[j] = norm(v)
            if β[j] < tol
                break
            end
            Q[:, j+1] = v ./ β[j]
        end

        Ts = diagm(0 => α[1:s], 1 => β[1:s-1], -1 => β[1:s-1])
        F = eigen(Ts)
        θ = real(F.values)
        U = F.vectors
        θ1 = maximum(θ)
        q1_new = Q[:, 1:s] * U[:, argmax(θ)]

        if abs(θ1 - θ1_prev) < tol
            return θ1, q1_new
        end
        θ1_prev = θ1
        q1 = q1_new
    end
    return θ1_prev, q1
end



# Generate a random Hermitian matrix for testing
using Random
Random.seed!(42)

# Generate a Hermitian matrix
n = 100
A = randn(n, n)
A = A + A'  # Ensure Hermitian

# Normalized random starting vector
q1 = randn(n)
q1 = normalize(q1)

# Compute largest eigenvalue
s = 20  # Number of Lanczos vectors before restarting
θ1, v1 = restarting_lanczos(A, q1, s)

# Ground truth
θ_true = maximum(eigen(A).values)

println("Restarting Lanczos result: ", θ1)
println("Exact largest eigenvalue: ", θ_true)
println("Error: ", abs(θ1 - θ_true))
