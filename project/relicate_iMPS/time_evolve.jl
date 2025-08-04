
function cost_function(circ, reg)
    ψ = statevec(reg |> circ)
    @show ψ[1]
    overlap = sqrt(abs2(ψ[1]))
    prob_00 = sum(abs2(ψ[i]) for i in 1:length(ψ) if (i-1) & 0b01100 == 0)
    #return -overlap
    return -prob_00
end

function train_TDVP_nograd(params, p, J, g, dt; maxiter=500, nbatch=1000)
    cost_history = Float64[]
    function objective(x)
       #circ = new_time_evolve_circ(x,p,J,g,dt)
       circ = time_evolve_circ(x, p,J,g,dt)
       reg = zero_state(nqubits(circ))
       cost = cost_function(circ, reg)
       push!(cost_history, cost)
       @info "p: $p, g: $g, Iter $(length(cost_history)), cost: $cost"
       return cost
    end

    @info "Number of parameters is $(length(params))"
  
    optimizer = NelderMead(; 
        parameters = Optim.AdaptiveParameters(),
        initial_simplex = Optim.AffineSimplexer()
    )
    result = optimize(objective, params, optimizer, Optim.Options(iterations=maxiter, show_trace=true))
    return cost_history
end

p=3
J = 1.0
g = 0.2
dt = 0.0
params = parameter1(p)
cost_history = train_TDVP_nograd(params, p, J, g, dt; maxiter=10000, nbatch=1000)


fig = Plots.plot(cost_history, xlabel="Iteration", ylabel="Cost", title="Training Cost vs. Iteration", legend=false)
Plots.savefig(fig, "cost_vs_iteration.png")
using Plots

using Random
Random.seed!(2)
parameter2(p)=rand(6+4p)
p=2
J = 1.0
g = 0.2
dt = 0.0
params = parameter2(p)
cost_history = train_TDVP_nograd(params, p, J, g, dt; maxiter=5000, nbatch=1000)