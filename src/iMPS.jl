# analitical result
function int(h::Float64, J::Float64)
    f(u,p) = sqrt((J-h)^2 + 4*J*h*sin(u/2)^2)/ (-2*π)
    domain= (-π,π)
    prob = IntegralProblem(f,domain)
    sol = solve(prob, HCubatureJL(); abstol=1e-10)
    return sol
end


# Result from MPSKit.jl 
function exact_energy(d,D,J,g)
    psi0 = InfiniteMPS([ℂ^d], [ℂ^D])
    H0 = transverse_field_ising(;J=J, g=g)
    psi,_= find_groundstate(psi0, H0, VUMPS())
    E = real(expectation_value(psi,H0))
    return psi, E
end

function exact_echo(d,D,J,g0,g1,dt)
    ψ₀,_ = exact_energy(d,D,J,g0)
    ψₜ = deepcopy(ψ₀)
    H₁ = transverse_field_ising(; J=J, g=g1)
    envs = environments(ψₜ, H₁)
    ψₜ, envs = timestep(ψₜ, H₁, 0, dt, TDVP(), envs);
    echo_ = echo(ψ₀, ψₜ)
    return envs
end
echo(ψ₀::InfiniteMPS, ψₜ::InfiniteMPS) =-2*log(abs(dot(ψ₀, ψₜ)))

function iMPS(d,D,J)
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    all=Float64[]
    for g in g_list
       _,energy = exact_energy(d,D,J,g)
       push!(all,energy)
    end
    return all
end

function Loschmit_echo(d,D,J,g0,g1)
    dt=collect(0.0:0.01:6.0)
    all=Float64[]
    for t in dt
        echo = exact_echo(d, D,J,g0,g1,t)
        push!(all,echo)
    end
    return all
end


# Channel iteration
function iterate_channel(gate, niters)
    rho1 = density_matrix(zero_state(1))  # the state qubit
    for i=1:niters
        rho2 = join(rho1, density_matrix(zero_state(1)))
        @assert all(iszero, measure(rho2, 1; nshots=1000))
        rho2 = apply!(rho2, gate)
        rho1 = partial_tr(rho2, 1) |> normalize!
    end
    return rho1
end
# domain eigen of transfer matrix
function exact_left_environment(gate)
    A = reshape(Matrix(gate), 2, 2, 2, 2)[:, :, 1, :]
    T = reshape(ein"iab,icd->cadb"(conj(A), A), 4, 4)
    @assert LinearAlgebra.eigen(T).values[end] ≈ 1.
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], 2, 2)
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    return rho
end

# local expectation value by exact contraction
function cost_X(gate; niters=100)
    A = reshape(Matrix(gate), 2, 2, 2, 2)[:,:,1,:]
    @assert ein"iab, iac->bc"(conj(A), A) ≈ I
    rho = iterate_channel(gate, niters)

    return ein"((ba, ica), jcb), ij->"(rho.state, conj(A), A, Matrix(X))[]
end
function cost_ZZ(gate; niters=100)
    A = reshape(Matrix(gate), 2, 2, 2, 2)[:,:,1,:]

    rho = iterate_channel(gate, niters)
    # cost = ein"ad,bca,be,efd,ghc,gj,jhf->"(rho.state, A_tensor, Matrix(Z),A_tensor_dagger,A_tensor,Matrix(Z),A_tensor_dagger)
    res = ein"((ba, ica), jdb), ij->cd"(rho.state, conj(A), A, Matrix(Z))
    return ein"((ab, ica), jcb), ij->"(res, conj(A), A, Matrix(Z))[]
end

# local expectation value by measurement
function cost_X_circ(gate; niters=100)
    rho = iterate_channel(gate, niters)

    rho2 = join(rho, density_matrix(zero_state(1)))
    @assert all(iszero, measure(rho2, 1; nshots=1000))
    rho2 = apply!(rho2, gate)
    apply!(rho2, put(2, 1=>H))

    return 1 - 2 * mean(measure(rho2, 1; nshots=50000))
end
function cost_ZZ_circ(gate; niters=100)
    rho = iterate_channel(gate, niters)

    rho2 = join(rho, density_matrix(zero_state(1)))
    @assert all(iszero, measure(rho2, 1; nshots=1000))
    rho2 = apply!(rho2, gate)

    return 1 - 2 * mean(measure(rho2, 1; nshots=50000))
end


function train_energy(params,g,J,p; maxiter=10000, nbatch=1000)
    X_history = Float64[]
    final_A = Matrix(I, 4,4)
    final_params = []
    function objective(x)
        A_matrix = Matrix(I,4,4)
        for r in 1:p
            gate = kron(Ry(x[2*r-1]), Ry(x[2*r]))
            cnot_12 = cnot(2,2,1)
            A_matrix *= Matrix(gate) * Matrix(cnot_12)
        end
        @assert A_matrix * A_matrix' ≈ I atol=1e-5
        @assert A_matrix' * A_matrix ≈ I atol=1e-5
        
        energy = -g*cost_X(matblock(A_matrix)) - J*cost_ZZ(matblock(A_matrix))
        #energy = -g*cost_X_circ(matblock(A_matrix)) - J*cost_ZZ_circ(matblock(A_matrix))^2
        push!(X_history, real(energy))
        @info "Iter $(length(X_history)), cost: $energy"
        final_A = A_matrix
        final_params = x
        return real(energy)
    end
    @info "Number of parameters is $(length(params))"
    optimizer = BFGS()
    Optim.optimize(objective, params, optimizer, Optim.Options(
        iterations=maxiter,
        show_trace=true,
        f_reltol = 1e-9,  
      
        g_tol = 1e-9,     
       
        time_limit=3600.0 
    ))
    return X_history, final_A, final_params
end


function all_energy(J=1.0)
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    p_values = [2,4,6]  # Different p values
    exact_energies = [int(g, J).u for g in g_list]
    
    # Calculate exact energies using exact_energy function for each g value
    exact_energies_mps = []
    exact_energies_mps_error = []
    for (g_idx, g) in enumerate(g_list)
        _, E_exact = exact_energy(2, 2, J, g)  # d=2, D=2
        error = max(abs(E_exact - exact_energies[g_idx]), 1e-15)
        push!(exact_energies_mps, E_exact)
        push!(exact_energies_mps_error, error)
        println("Exact energy (MPS) for g=$g: $E_exact")
    end
    
    all_energy_lists = []
    
    # Create data directory if it doesn't exist
    if !isdir("data")
        mkdir("data")
    end
    
    # Save exact MPS energies to independent file
    open("data/exact_energies_mps.dat", "w") do file
        write(file, "# g_value\texact_energy_int\texact_energy_mps\n")
        for (g_idx, g) in enumerate(g_list)
            write(file, "$(g)\t$(exact_energies_mps[g_idx])\t$(exact_energies_mps_error[g_idx])\n")
        end
    end
    println("Exact MPS energies saved to data/exact_energies_mps.dat")
    
    # Create separate file for each p value
    for (p_idx, p_val) in enumerate(p_values)
        filename = "data/energy_results_p=$(p_val).dat"
        energy_list = Float64[]
        
        open(filename, "w") do file
            # Write header
            write(file, "# g_value\tenergy\terror\texact_energy\n")
            
            # Calculate energies for each g value
            for (g_idx, g) in enumerate(g_list)
                params = rand(6*p_val)
                energy_history, _, _ = train_energy(params, g, J, p_val)
                energy = energy_history[end]  # Get final energy
                error = max(abs(energy - exact_energies[g_idx]), 1e-15)
                
                # Store energy values
                push!(energy_list, energy)
                
                # Write line to file
                write(file, "$(g)\t$(energy)\t$(error)\t$(exact_energies[g_idx])\n")
                
                println("p=$p_val, g=$g: energy=$(energy), error=$(error)")
            end
        end
        
        push!(all_energy_lists, energy_list)
        println("Data for p=$p_val saved to $filename")
    end
    
    return all_energy_lists
end

function draw_figure()
    # Read data from .dat files
    p_values = [2, 4, 6]
    colors = [RGBA(1,0,0,0.5), RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), RGBA(0,0,1,0.5)]
    markers = [:+, :x, :star,:diamond]
    
    # Create the plot
    fig = Plots.plot(xlabel="g", ylabel="ε - ε_exact", title="Energy Error vs Transverse Field", 
                     ylims=(1e-15, 10), xlims=(-0.25, 2), 
                     yscale=:log10, 
                     yticks=[1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1],
                     xticks=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    
    # Plot data for each p value
    for (p_idx, p_val) in enumerate(p_values)
        filename = "data/energy_results_p=$(p_val).dat"
        
        if isfile(filename)
            # Read the data file
            g_values = Float64[]
            errors = Float64[]
            
            open(filename, "r") do file
                for line in eachline(file)
                    # Skip empty lines and comments
                    if !isempty(strip(line)) && !startswith(line, "#")
                        # Try splitting by tab first, then by spaces
                        parts_tab = split(line, "\t")
                        parts_space = split(line)
                        
                        # Use the split that gives us at least 4 parts
                        parts = length(parts_tab) >= 4 ? parts_tab : parts_space
                        
                        if length(parts) >= 4
                            try
                                g = parse(Float64, parts[1])
                                error = parse(Float64, parts[3])  # error column
                                push!(g_values, g)
                                push!(errors, error)
                            catch e
                                println("Warning: Could not parse line: $line")
                                println("Error: $e")
                            end
                        end
                    end
                end
            end
            
            # Plot the data
            if !isempty(g_values)
                Plots.plot!(fig, g_values, errors,
                    label="p=$p_val, $(2*p_val) parameters",
                    color=colors[p_idx],
                    marker=markers[p_idx],
                    markersize=4,
                    linewidth=2)
                Plots.plot!(fig, legend=:bottomright, legendfontsize=10)
            end
            
            println("Plotted data for p=$p_val from $filename")
        else
            println("Warning: File $filename not found")
        end
    end
    
    # Add exact MPS comparison if available
    mps_filename = "data/exact_energies_mps.dat"
    if isfile(mps_filename)
        g_values_mps = Float64[]
        errors_mps = Float64[]
        
        open(mps_filename, "r") do file
            for line in eachline(file)
                if !isempty(strip(line)) && !startswith(line, "#")
                    parts = split(line, "\t")
                    if length(parts) >= 3
                        g = parse(Float64, parts[1])
                        error = parse(Float64, parts[3])  # MPS vs int error
                        push!(g_values_mps, g)
                        push!(errors_mps, error)
                    end
                end
            end
        end
        
        if !isempty(g_values_mps)
            Plots.plot!(fig, g_values_mps, errors_mps,
                label="iMPS",
                color=colors[4],
                marker=markers[4],
                markersize=4,
                linewidth=2,
                linestyle=:dash)
            println("Added MPS vs Integration comparison")
        end
    end
    
    # Save and display the plot
    Plots.savefig(fig, "data/energy_error_plot.png")
    Plots.display(fig)
    
    println("Plot saved as data/energy_error_plot.png")
    return fig
end

draw_figure()





function get_gate(params,niters)
    gate_y1 = kron(Rx(params[1]), Rx(params[2]))
    gate_y2 = kron(Rz(params[3]), Rz(params[4]))
    gate_y3 = kron(Rx(params[5]), Rx(params[6]))
    gate_y4 = kron(Rz(params[7]), Rz(params[8]))
    gate_y5 = kron(Rx(params[9]), Rx(params[10]))
    gate_y6 = kron(Rz(params[11]), Rz(params[12]))
    cnot_12 = cnot(2,1,2)
     
    A_matrix = mat(gate_y1)*Matrix(cnot_12)*mat(gate_y2)*Matrix(cnot_12)*mat(gate_y3)*Matrix(cnot_12)*mat(gate_y4)*Matrix(cnot_12)*mat(gate_y5)*Matrix(cnot_12)*mat(gate_y6)* Matrix(cnot_12)
    A = matblock(A_matrix)
    #=
    rho = iterate_channel(A,niters).state
    L = cholesky(Hermitian(rho)).L
    L_vec = reshape(L, 4, 1)  # Flatten 2×2 → 4×1
    L_vec = L_vec / norm(L_vec)
    remaining = nullspace(L_vec')
    V = hcat(L_vec, remaining)=#
    
    #@assert V*V' ≈ I && V'*V ≈ I
    #@assert V[:,1]==L_vec
    return A
end

function iterate_circuit(params,niters)
    nbit = 4
    circ = chain(nbit)
    A,V = get_gate(params,niters)
    push!(circ, put(nbit,(3,4)=>matblock(V)))
    push!(circ, put(nbit,(2,3)=>A))
    push!(circ, put(nbit,(1,3)=>A))
    #push!(circ, cnot(nbit,1,2)) 
    #push!(circ, Measure(nbit; locs=[1,2]))

    return circ
end

function iterate_circuit_x(params,niters)
    nbit = 4
    circ = chain(nbit)
    nbit = 4
    circ = chain(nbit)
    A,V = get_gate(params,niters)
    push!(circ, put(nbit,(3,4)=>matblock(V)))
    push!(circ, put(nbit,(2,3)=>A))
    push!(circ, put(nbit,(1,3)=>A))
    push!(circ, put(nbit,1=>H))
    push!(circ, put(nbit,2=>H))
    #push!(circ, Measure(nbit; locs=[1,2]))

    return circ
end


function cost_iter(circ0,circ1, reg0,reg1,J,g)
     #=
    res_z = gensample(circ0, copy(reg0), Yao.Z)
    cost_z = -J*mean((-1).^res_z[:,1] .* (-1).^res_z[:,2])
    res_x = gensample(circ1, copy(reg1), Yao.Z)
    cost_x = -0.5*g*(mean((-1).^res_x[:,1])+mean((-1).^res_x[:,2]))
    energy = cost_z+cost_x
    @show cost_z,cost_x=#
   
    ψ0 = statevec(copy(reg0) |> circ0)
    prob_01_or_10 = sum(abs2(ψ0[i]) for i in 1:length(ψ0) 
                        if (((i-1) >> 0) & 1) != (((i-1) >> 1) & 1))
    prob_00_or_11 = 1.0 - prob_01_or_10
    

    ψ1 = statevec(copy(reg1) |> circ1)
    prob_01_or_10_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
    if (((i-1) >> 0) & 1) != (((i-1) >> 1) & 1))
    prob_00_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
    if (((i-1) >> 0) & 1) == 0 && (((i-1) >> 1) & 1) == 0)
    prob_11_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
    if (((i-1) >> 0) & 1) == 1 && (((i-1) >> 1) & 1) == 1)
    prob_00_or_11_x = 1.0 - prob_01_or_10_x   
  

    prob_00_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
              if ((i-1) >> 0) & 1 == 0 && ((i-1) >> 1) & 1 == 0)
    prob_01_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
              if ((i-1) >> 0) & 1 == 0 && ((i-1) >> 1) & 1 == 1)
    prob_10_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
              if ((i-1) >> 0) & 1 == 1 && ((i-1) >> 1) & 1 == 0)
    prob_11_x = sum(abs2(ψ1[i]) for i in 1:length(ψ1) 
              if ((i-1) >> 0) & 1 == 1 && ((i-1) >> 1) & 1 == 1)

    @show abs2(prob_00_x + prob_01_x + prob_10_x + prob_11_x - 1.0) 
    @show prob_00_or_11, prob_01_or_10
    @show prob_00_x, prob_11_x
    energy = -J*(prob_00_or_11-prob_01_or_10)-g*(prob_00_x-prob_11_x)
    return energy
end


function train_with_gradient(params, g, niters; maxiter=500, optimizer=Optimisers.ADAM(0.01))
    J=1.0
    circ0 = iterate_circuit(params, niters)
    circ1 = iterate_circuit_x(params, niters)
    reg0 = zero_state(nqubits(circ0))
    reg1 = zero_state(nqubits(circ1))

    energy_history = Float64[]
    opt = Optimisers.setup(optimizer, params)
    @info "Number of parameters is $(length(params))"
    for i in 1:maxiter
        grad = get_gradient(params, J, g, niters)
        Optimisers.update!(opt, params, grad)
        circ0 = iterate_circuit(params,niters)
        circ1 = iterate_circuit_x(params,niters) 
        push!(energy_history, cost_iter(circ0, circ1, reg0, reg1, J, g))
        @info "niters: $niters, g: $g, Iter $i, Energy: $(energy_history[end])"
        @show grad
    end
    return energy_history
end

function train_iter_circ(params,g,niters; maxiter=3000, nbatch=1000)
    energy_history = Float64[]
    J=1.0
    function objective(x)
        circ0 = iterate_circuit(x,niters)
        circ1 = iterate_circuit_x(x,niters)
        reg0 = join(zero_state(nqubits(circ0)))
        reg1 = join(zero_state(nqubits(circ1)))
        energy = cost_iter(circ0, circ1, reg0, reg1,J,g)
        push!(energy_history, energy)
        @info "Iter $(length(energy_history)), cost: $energy"
        return energy
    end
    @info "Number of parameters is $(length(params))"
    optimizer = NelderMead(; 
        parameters = Optim.AdaptiveParameters(),
        initial_simplex = Optim.AffineSimplexer()
    )
    result = Optim.optimize(objective, params, optimizer, Optim.Options(
        iterations=maxiter,
        show_trace=true,
        f_tol=1e-12,        
        g_tol=1e-10,        
        x_tol=1e-12,       
        f_abstol=-1.015687012,     
        time_limit=3600.0  
    ))
    @info "Converged: $(Optim.converged(result))"
    params .= result.minimizer
    return energy_history
end

function train_iter_channel(params,g,niters; maxiter=3000)
    energy_history = Float64[]
    J=1.0
    function objective(x)
        A = get_gate(x,niters)
        rho = iterate_channel(A,niters)
        energy = -J*(expect(put(1,1=>Z),rho)^2) - g*expect(put(1,1=>X),rho)
        @show J*(expect(put(1,1=>Z),rho)^2), g*expect(put(1,1=>X),rho)
        push!(energy_history, energy)
        @info "Iter $(length(energy_history)), cost: $energy"
        return energy
    end
    @info "Number of parameters is $(length(params))"
    optimizer = NelderMead(; 
        parameters = Optim.AdaptiveParameters(),
        initial_simplex = Optim.AffineSimplexer()
    )
    result = Optim.optimize(objective, params, optimizer, Optim.Options(
        iterations=maxiter,
        show_trace=true,
        f_tol=1e-12,        
        g_tol=1e-10,        
        x_tol=1e-12,       
    ))
    @info "Converged: $(Optim.converged(result))"
    params .= result.minimizer
    return energy_history
end

function compare_with_MPSKit(params,g,niters)
    # iMPS by MPSKit
    D = 2
    psi, E = exact_energy(2,D,J,g);
    exact_A = Array{ComplexF64}(undef, D, 2, D)
    for (i, j, k) in Iterators.product(1:D, 1:2, 1:D)
        exact_A[i, j, k] = psi.AR.data[1][i, j, k]
    end
    @assert ein"aib, cib->ac"(conj(exact_A), exact_A) ≈ I

    # Isometry by VUMPS
    exact_V = reshape(exact_A, size(exact_A)[1], :)
    @assert exact_V * exact_V' ≈ I
    # Unitary by VUMPS
    exact_U = vcat(exact_V[1:1,:], nullspace(exact_V)'[1:1,:], exact_V[2:2,:], nullspace(exact_V)'[2:2,:])
    @assert exact_U * exact_U' ≈ I
    @assert exact_U' * exact_U ≈ I

    exact_gate = transpose(exact_U);
    -g*cost_X(matblock(exact_gate);niters=1000) - J*cost_ZZ(matblock(exact_gate);niters=1000)
    -g*cost_X_circ(matblock(exact_gate);niters=1000) - J*cost_ZZ(matblock(exact_gate);niters=1000)
    int(g,J).u
end

using TensorKit, MPSKit, MPSKitModels
using LinearAlgebra,Yao,OMEinsum,Optim,Statistics,Plots
using Integrals

psi, E = exact_energy(2,2,1.0,0.25)
@show psi
env = exact_echo(2,2,1.0,0.25,0.25,1.0)


gate_z = kron(Rz(pi/4), Rz(pi/4))
gate_x = kron(Rx(pi/4), Rx(pi/4))
cnot_12 = cnot(2,1,2)
A_matrix = mat(gate_z) * mat(gate_x) * Matrix(cnot_12)  
A_matrix = rand_unitary(ComplexF64, 4)

# iterate channel, exact contraction
p=6
params =  rand(2*p)
g = 0.25
J =1.0
energy, final_A, final_params = train_energy(params,g,J,p)

params
final_params

params .- final_params
min_energy = minimum(energy)
error = min_energy-int(g,J).u 
fig = Plots.plot(energy, xlabel="Iteration", ylabel="Cost", title="Training Cost vs. Iteration", legend=false)



# circuit without gradient
p=6
params = rand(2*p)
g=0.0
energy = train_iter_circ(params,g,100)
min_energy = minimum(energy)
error = minimum(energy)-int(g,1.0).u
mean(energy[end-20:end])
fig = Plots.plot(energy, xlabel="Iteration", ylabel="Cost", title="Training Cost vs. Iteration", legend=false)


# iterate channel, Cholesky decomposition
p = 6
params = randn(2*p)
g = 2.0
niters = 50
energy_history = train_iter_channel(params, g, niters)
min_energy = minimum(energy_history)
error = min_energy-int(g,1.0).u
fig = Plots.plot(energy_history, xlabel="Iteration", ylabel="Cost", title="Training Cost vs. Iteration", legend=false)


