
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
 

function cost_X(A_matrix)
    A = matblock(A_matrix)
    A_tensor = reshape(A_matrix, 2, 2, 2, 2)[:,1,:,:]
    A_tensor_dagger = reshape(reshape(conj(A_tensor),2,4),2,2,2)
    rho = iterate_channel(A, 100)
    cost = ein"ad,abc,cf,dbf->"(rho.state, A_tensor, Matrix(X),A_tensor_dagger)
end

function cost_ZZ(A_matrix)
    A = matblock(A_matrix)
    A_tensor = reshape(A_matrix, 2, 2, 2, 2)[:,1,:,:]
    A_tensor_dagger = reshape(reshape(conj(A_tensor),2,4),2,2,2)
    rho = iterate_channel(A, 100)
    cost = ein"ad,abc,cf,def,bgh,hj,egj->"(rho.state, A_tensor, Matrix(Z),A_tensor_dagger,A_tensor,Matrix(Z),A_tensor_dagger)
end

function train_energy(params,g,p; maxiter=1000, nbatch=1000)
    X_history = Float64[]
    function objective(x)
        A_matrix = Matrix(I,4,4)
        for r in 1:p
            gate_x = kron(Rz(x[1+6*(r-1)]), Rz(x[2+6*(r-1)]))
            gate_z = kron(Rx(x[3+6*(r-1)]), Rx(x[4+6*(r-1)]))
            gate_x2 = kron(Rz(x[5+6*(r-1)]), Rz(x[6+6*(r-1)]))
            cnot_12 = cnot(2,1,2)
            A_matrix *= mat(gate_x) * mat(gate_z) * mat(gate_x2) * Matrix(cnot_12)
        end
        energy = -g*cost_X(A_matrix)[]-cost_ZZ(A_matrix)[]
        @show cost_X(A_matrix)[]
        @show cost_ZZ(A_matrix)[]
        push!(X_history, real(energy))
        @info "Iter $(length(X_history)), cost: $energy"
        return real(energy)
    end
    @info "Number of parameters is $(length(params))"
    optimizer = NelderMead(; 
        parameters = Optim.AdaptiveParameters(),
        initial_simplex = Optim.AffineSimplexer()
    )
    Optim.optimize(objective, params, optimizer, Optim.Options(
        iterations=maxiter,
        show_trace=true,
        f_tol=1e-8,        # Stop if function change < 1e-8
        g_tol=1e-6,        # Stop if gradient norm < 1e-6  
        x_tol=1e-8,        # Stop if parameter change < 1e-8
        f_abstol=-Inf,     # Stop if function value < threshold (set to desired energy)
        time_limit=3600.0  # Stop after 1 hour (3600 seconds)
    ))
    return X_history
end

function all_energy(p)
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    energy_list = Float64[]
    exact_energies = [int(g,1.0).u for g in g_list]
    fig = Plots.plot(xlabel="g", ylabel="ε - ε_exact", title="Energy Error vs Transverse Field", 
                     ylims=(1e-15, 10), xlims=(-0.25, 2), 
                     yscale=:log10, 
                     yticks=[1e-15,1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1,1e1],
                     xticks=[-0.25,0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    for (g_idx, g) in enumerate(g_list)
        params = rand(6*p)
        energy = train_energy(params,g,p)
        push!(energy_list, energy)
        
    end
    errors = max.(abs.(energy_list-exact_energies), 1e-15)
    Plots.plot!(fig, g_list, errors, 
        label="p=1, 6 parameters",
        color="red", 
        marker="o",
        markersize=4,
        linewidth=2)
    Plots.savefig(fig, "classical.png")
    Plots.display(fig)
    return energy_list
end

energy_list=all_energy(4)
println("Complete energy_list with all 9 elements:")
for (i, energy) in enumerate(energy_list)
    println("  [$i]: $energy")
end

function iterate_circuit(params,niters)
    nbit = 4
    circ = chain(nbit)
    A_matrix = Matrix(I,4,4)
    for r in 1:p
        gate_z = kron(Ry(params[1+2*(r-1)]), Ry(params[2+2*(r-1)]))
        #gate_x = kron(Rx(params[3+6*(r-1)]), Rx(params[4+6*(r-1)]))
        #gate_z2 = kron(Rz(params[5+6*(r-1)]), Rz(params[6+6*(r-1)]))
        cnot_12 = cnot(2,2,1)
        A_matrix *= mat(gate_z) * Matrix(cnot_12)
    end
    A = matblock(A_matrix)
    V = iterate_channel(A,niters).state
    V = reshape(V,4,1)
    remaining = nullspace(V') 
    V = hcat(V, remaining)

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
    A_matrix = Matrix(I,4,4)
    for r in 1:p
        gate_z = kron(Ry(params[1+2*(r-1)]), Ry(params[2+2*(r-1)]))
        #gate_x = kron(Rx(params[3+6*(r-1)]), Rx(params[4+6*(r-1)]))
        #gate_z2 = kron(Rz(params[5+6*(r-1)]), Rz(params[6+6*(r-1)]))
        cnot_12 = cnot(2,2,1)
        A_matrix *= mat(gate_z) * Matrix(cnot_12)
    end
    A = matblock(A_matrix)
    V = iterate_channel(A,niters).state
    V = reshape(V,4,1)
    remaining = nullspace(V') 
    V = hcat(V, remaining)

    push!(circ, put(nbit,(3,4)=>matblock(V)))
    push!(circ, put(nbit,(2,3)=>A))
    push!(circ, put(nbit,(1,3)=>A))
    #push!(circ, cnot(nbit,1,2)) 
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
    energy = -J*(prob_00_or_11-prob_01_or_10)-g*(prob_11_x-prob_00_x)
    @show prob_01_or_10,prob_00_or_11
    @show prob_00_x,prob_11_x     
    return energy
end

function train_iter_circ(params,g,niters; maxiter=3000, nbatch=1000)
    energy_history = Float64[]
    J=1.0
    function objective(x)
        circ0 = iterate_circuit(x,niters)
        circ1 = iterate_circuit_x(x,niters)
        reg0 = zero_state(nqubits(circ0))
        reg1 = zero_state(nqubits(circ1))
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
        f_abstol=-1.0156870128527515,     
        time_limit=3600.0  
    ))
    @info "Converged: $(Optim.converged(result))"
    params .= result.minimizer
    return energy_history
end


using TensorKit, MPSKit, MPSKitModels
using LinearAlgebra,Yao,OMEinsum,Optim,Statistics,Plots

gate_z = kron(Rz(pi/4), Rz(pi/4))
gate_x = kron(Rx(pi/4), Rx(pi/4))
cnot_12 = cnot(2,1,2)
A_matrix = mat(gate_z) * mat(gate_x) * Matrix(cnot_12)  
A_matrix = rand_unitary(ComplexF64, 4)

# try exact contraction
p=5
params = rand(6*p)
g=0.5
energy = train_energy(params,g,p)
min_energy = minimum(energy)
mean(energy[end-20:end])
fig = Plots.plot(energy, xlabel="Iteration", ylabel="Cost", title="Training Cost vs. Iteration", legend=false)

# try circuit
p=5
params = rand(2*p)
g=0.5
energy = train_iter_circ(params,g,50)
min_energy = minimum(energy)-int(g,1.0).u
mean(energy[end-20:end])
fig = Plots.plot(energy, xlabel="Iteration", ylabel="Cost", title="Training Cost vs. Iteration", legend=false)





e = eigen(Hermitian(state.state)).values

ψ₁ = eigen(Hermitian(state.state)).vectors[:, 1]
ψ₁ /= LinearAlgebra.norm(ψ₁)
 
ψ₂ = eigen(Hermitian(state.state)).vectors[:, 2]
ψ₂ /= LinearAlgebra.norm(ψ₂)

e[1] * LinearAlgebra.norm(ψ₁[1])^2 + e[2] * LinearAlgebra.norm(ψ₂[1])^2

state.state