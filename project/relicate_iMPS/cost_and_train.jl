
function cost_gs_energy(circ0, circ1, reg0,reg1, J, g)
    energy = 0.0
    #=
    res_z = gensample(circ0, reg0, Yao.Z)
    energy -= J*mean((-1).^res_z[:,1] .* (-1).^res_z[:,2])
    res_x = gensample(circ1, reg0, Yao.Z)
    energy -= 0.5*g*(mean((-1).^res_x[:,1])+mean((-1).^res_x[:,2]))
    =#
    ψ0 = statevec(copy(reg0) |> circ0)
    @show sum((ψ0[i]) for i in 1:length(ψ0))
    prob_01_or_10 = sum(abs2(ψ0[i]) for i in 1:length(ψ0) 
                        if (((i-1) >> 2) & 1) != (((i-1) >> 3) & 1))
    prob_00_or_11 = sum(abs2(ψ0[i]) for i in 1:length(ψ0) 
    if (((i-1) >> 2) & 1) == 0 && (((i-1) >> 3) & 1) == 0)
    @show prob_01_or_10,prob_00_or_11

    ψ1 = statevec(copy(reg1) |> circ1)
    prob_01_or_10_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
    if (((i-1) >> 2) & 1) != (((i-1) >> 3) & 1))
    prob_00_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
    if (((i-1) >> 2) & 1) == 0 && (((i-1) >> 3) & 1) == 0)
    prob_11_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
    if (((i-1) >> 2) & 1) == 1 && (((i-1) >> 3) & 1) == 1)
    @show prob_01_or_10_x,prob_00_x,prob_11_x
    energy = -J*(prob_00_or_11-prob_01_or_10)-g*(prob_11_x-prob_00_x)
    return energy
end

function cost_env(circ2, circ3, circ4, reg2,reg3,reg4)
    trace_dis = 0.0
    #=
    res1 = gensample(circ1, reg1, Yao.Z)
    res2 = gensample(circ2, reg2, Yao.Z)
    res3 = gensample(circ3, reg3, Yao.Z)
    total = nbatch(reg1)
    rr = 1-2*count(row -> row == [1,1], eachrow(res1))/total
    rs = 1-2*count(row -> row == [1,1], eachrow(res2))/total
    ss = 1-2*count(row -> row == [1,1], eachrow(res3))/total
    trace_dis += rr+ss-2*rs=#

    psi1 = statevec(copy(reg2) |> circ2)
    psi2 = statevec(copy(reg3) |> circ3)
    psi3 = statevec(copy(reg4) |> circ4)

    prob1_00 = sum(abs2(psi1[i]) for i in 1:length(psi1) 
    if (((i-1) >> 2) & 1) == 0 && (((i-1) >> 5) & 1) == 0)

    prob2_00 = sum(abs2(psi2[i]) for i in 1:length(psi2) 
    if (((i-1) >> 2) & 1) == 0 && (((i-1) >> 4) & 1) == 0)

    prob3_00 = sum(abs2(psi3[i]) for i in 1:length(psi3) 
    if (((i-1) >> 1) & 1) == 0 && (((i-1) >> 3) & 1) == 0)

    r_square = 1-2*prob1_00
    r_s = 1-2*prob2_00
    s_square = 1-2*prob3_00
    trace_dis = abs(r_square+s_square-2*r_s)
    return trace_dis
end
function cost(circ0, circ1, circ2, circ3, circ4, reg0,reg1,reg2,reg3,reg4, J, g)
    energy = cost_gs_energy(circ0, circ1, reg0, reg1,J, g)
    trace_dis = cost_env(circ2, circ3, circ4, reg2,reg3,reg4)
    total_cost = energy + trace_dis
    return total_cost
end
 
function gradient_parameter_shift(params, p, J, g; nbatch=1000)
    nparams = length(params)
    grad = zeros(Float64, nparams)
    
    for i in 1:nparams
        params_plus = copy(params)
        params_plus[i] += π/2
        
        params_minus = copy(params)
        params_minus[i] -= π/2

        circ0_plus = ground_state_circ(params_plus,p)
        circ0_minus = ground_state_circ(params_minus,p)
        circ1_plus = ground_state_circ_x(params_plus,p)
        circ1_minus = ground_state_circ_x(params_minus,p)

        reg0_plus = zero_state(nqubits(circ0_plus))
        reg0_minus = zero_state(nqubits(circ0_minus))

        cost_plus = cost_gs_energy(circ0_plus, circ1_plus, reg0_plus, J, g)
        cost_minus = cost_gs_energy(circ0_minus, circ1_minus, reg0_minus, J, g)
 
        grad[i] = 0.5 * (cost_plus - cost_minus)
    end
    
    return grad
end
 
 
function train(params,p,g; maxiter=500, optimizer=Optimisers.ADAM(0.01), nbatch=1000)
    J=1.0
    circ0 = ground_state_circ(params,p)
    circ1 = ground_state_circ_x(params,p)
    reg0 = zero_state(nqubits(circ0);nbatch=1000)

    energy_history = Float64[]
    opt = Optimisers.setup(optimizer, params)
    @info "Number of parameters is $(length(params))"
    for i in 1:maxiter
        grad = gradient_parameter_shift(params, p, J, g; nbatch=nbatch)
        Optimisers.update!(opt, params, grad)
        circ0 = ground_state_circ(params,p)
        circ1 = ground_state_circ_x(params,p) 
        push!(energy_history, cost_gs_energy(circ0, circ1, reg0, J, g))
        @info "p: $p, g: $g, Iter $i,Energy: $(energy_history[end])"
        @show grad
    end
    return energy_history
end

function train_env(params,p,g; maxiter=500, optimizer=Optimisers.ADAM(0.01), nbatch=1000)
    J=1.0
    env_cost_history = Float64[]

    function objective(x)
        circ2 = env_circ1(x,p)
        circ3 = env_circ2(x,p)
        circ4 = env_circ3(x,p)
        reg2 = zero_state(nqubits(circ2))
        reg3 = zero_state(nqubits(circ3))
        reg4 = zero_state(nqubits(circ4))
        env_cost = cost_env(circ2, circ3, circ4, reg2,reg3,reg4)
        push!(env_cost_history, env_cost)
        @info "p: $p, g: $g, Iter $(length(env_cost_history)), cost: $env_cost"
        return env_cost
    end
    optimizer = NelderMead(; 
        parameters = Optim.AdaptiveParameters(),
        initial_simplex = Optim.AffineSimplexer()
    )
    result = optimize(objective, params, optimizer, Optim.Options(iterations=maxiter, show_trace=true))
    @info "Optimization completed. Final energy: $(result.minimum)"
    @info "Converged: $(Optim.converged(result))"
    params .= result.minimizer
    return params
end


function train_no_grad(params, p, g; maxiter=2000, nbatch=1000)
    J = 1.0
    energy_history = Float64[]
    
    function objective(x)
        circ0 = ground_state_circ(x, p)
        circ1 = ground_state_circ_x(x, p)
        circ2 = env_circ1(x,p)
        circ3 = env_circ2(x,p)
        circ4 = env_circ3(x,p)
        reg0 = zero_state(nqubits(circ0))
        reg1 = zero_state(nqubits(circ1))
        reg2 = zero_state(nqubits(circ2))
        reg3 = zero_state(nqubits(circ3))
        reg4 = zero_state(nqubits(circ4))
        total_cost = cost(circ0, circ1, circ2, circ3, circ4, reg0,reg1,reg2,reg3,reg4, J, g)
        energy = cost_gs_energy(circ0, circ1, reg0, reg1, J, g)
        @show energy
        @show total_cost
        push!(energy_history, energy)
        @info "p: $p, g: $g, Iter $(length(energy_history)), Energy: $energy"
        return total_cost
    end
    
    @info "Number of parameters is $(length(params))"
  
    optimizer = NelderMead(; 
        parameters = Optim.AdaptiveParameters(),
        initial_simplex = Optim.AffineSimplexer()
    )
    
    # Run optimization
    result = Optim.optimize(objective, params, optimizer, Optim.Options(
        iterations=maxiter,
        show_trace=true,
        f_tol=1e-12,        
        g_tol=1e-10,       
        x_tol=1e-12,       
        f_abstol=-1.0156870128527515,     
        time_limit=3600.0  
    ))
    
    @info "Optimization completed. Final energy: $(result.minimum)"
    @info "Converged: $(Optim.converged(result))"
    
    params .= result.minimizer
    
    return energy_history
end

function draw_figure()
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    p_list = [1,2,3]
    n = 10 
    exact_energies = [int(g,1.0).u for g in g_list]

    fig = Plots.plot(xlabel="g", ylabel="ε - ε_exact", title="Energy Error vs Transverse Field",xlims=(0, 2))
    colors = [:red, :green, :orange]
    markers = [:+, :x, :star]

    for (idx, p) in enumerate(p_list)
        vqe_energies = Float64[]
        vqe_energies_filename = "vqe_energies_p$(p).txt"
        f = open(vqe_energies_filename, "w")
        println(f, "p,g,vqe_energy")
        errors = Float64[]   
        for (g_idx, g) in enumerate(g_list)
            params = parameter1(p)
            energy_history = train(params, p, g; maxiter=500, nbatch=4000)
            vqe_energy = mean(energy_history[450:500]) 
            
            push!(vqe_energies, vqe_energy)
            error = abs(vqe_energy - exact_energies[g_idx])
            push!(errors, max(error, 1e-13))  
            println(f, "$p,$g, $vqe_energy")
        end 
        close(f)
        Plots.plot!(fig, g_list, errors, 
        label="p=$p, $(3+2*p) parameters",
        color=colors[idx], 
        marker=markers[idx],
        markersize=4,
        linewidth=2)
    end
    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:topright, legendfontsize=10)
    Plots.savefig(fig, "energy_errorvs_g.png")
    Plots.display(fig)
    return fig
end

function draw_figure2()
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    p_list = [1,2,3]
    p1 = [-0.9999999997563389, -1.0156249970365168, -1.0624999968244235, -1.1406249987413477, -1.2499999965650135, -1.3906249993676547, -1.5624999981344312, -1.7656249994209183, -1.9999999510714537]
    p2 = [ -0.9999999991992583, -1.0156268712491878, -1.0624538461763937, -1.1407879533141116, -1.2543680873132481, -1.3910471626247445, -1.5787566107430402, -1.818739684731014, -2.0981094737626123]
    p3 = [ -0.9999999999999999, -1.0156879377212527, -1.063531806128337, -1.1465848784797186, -1.2725840951600263, -1.457071026676012, -1.6720898810656172, -1.8969280500308125, -2.1270280883760127]
    iMPS = [-0.9999999999999999,-1.0156870118440362,-1.0635440740663125,-1.1464647915423063,-1.2725424859373677,-1.459172235772456,-1.6717366238936089,-1.8959687944364405,-2.1270514348502796]
    exact_energies = [int(g,1.0).u for g in g_list]
    
    fig = Plots.plot(xlabel="g", ylabel="ε - ε_exact", title="Energy Error vs Transverse Field", 
                     ylims=(1e-15, 10), xlims=(-0.25, 2), 
                     yscale=:log10, 
                     yticks=[1e-15,1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1,1e1],
                     xticks=[-0.25,0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    colors = [:red, :green, :orange,:blue]
    markers = [:+, :x, :star,:diamond]

   
    errors1 = max.(abs.(p1-exact_energies), 1e-15)
    errors2 = max.(abs.(p2-exact_energies), 1e-15)
    errors3 = max.(abs.(p3-exact_energies), 1e-15)
    errors4 = max.(abs.(iMPS-exact_energies), 1e-15)
    Plots.plot!(fig, g_list, errors1, label="p=1, $(4) parameters",color=colors[1], marker=markers[1],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, errors2, label="p=2, $(8) parameters",color=colors[2], marker=markers[2],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, errors3, label="p=3, $(12) parameters",color=colors[3], marker=markers[3],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, errors4, label="iMPS",color=colors[4], marker=markers[4],markersize=4,linewidth=2)

    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:bottomright, legendfontsize=10)
    Plots.savefig(fig, "energy_errorvs_g.png")
    Plots.display(fig)
end

function int(h::Float64, J::Float64)
    f(u,p) = sqrt((J-h)^2 + 4*J*h*sin(u/2)^2)/ (-2*π)
    domain= (-π,π)
    prob = IntegralProblem(f,domain)
    sol = solve(prob, HCubatureJL(); abstol=1e-10)
    return sol
end


using Yao
using Optimisers
using Test
using Statistics
using Integrals
using Plots
using Optim

draw_figure2()

p=3;g=0.25
params = parameter1(p)
energy = train(params,p,g)
min_energy = minimum(energy)
mean(energy[end-50:end])
fig = Plots.plot(energy, xlabel="Iteration", ylabel="Cost", title="Training Cost vs. Iteration", legend=false)

Plots.savefig(fig, "cost_vs_iteration.png")

params = parameter1(p) 
energy = train_no_grad(params, p, g; maxiter=2000, nbatch=1000)
min_energy = minimum(energy)-int(g,1.0).u
