function iterate_channel_PEPS(gate, row; niters=200)
    rho = density_matrix(zero_state(row+1))
    for i in 1:niters
        for j in 1:row
            rho_p = density_matrix(zero_state(1))
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>gate))       
            rho = partial_tr(rho, 1) |> normalize!
        end
        #@info "eigenvalues of ρ = " eigen(Hermitian(rho.state)).values
        #@show isapprox(rho.state, rho.state') 
    end
    return rho
end

function exact_left_eigen(gate, nsites)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    #E = reshape(ein"iabcd,iefgh -> abefcdgh"(A, conj(A)), 4,4,4,4)
    _, T = contract_Elist([A for _ in 1:nsites], [conj(A) for _ in 1:nsites], nsites)
    T = reshape(T, 4^(nsites+1), 4^(nsites+1))
    #λ₁ = partialsort(abs.(LinearAlgebra.eigen(T).values),2;rev=true)
    λ₁ = LinearAlgebra.eigen(T).values[end-1]
    @show LinearAlgebra.eigen(T).values
    @show λ₁
    gap = -log(abs(λ₁))
    @assert LinearAlgebra.eigen(T).values[end] ≈ 1.
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^(nsites+1))), Int(sqrt(4^(nsites+1))))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    return rho, gap
end

# contract list of A and A^dagger to form transfer matrix  
function contract_Elist(tensor_ket, tensor_bra, row; optimizer=GreedyMethod())
    store = IndexStore()
    index_bra = Vector{Int}[]
    index_ket = Vector{Int}[]
    index_output = Int[]
    first_down_ket = newindex!(store)
    first_up_ket = newindex!(store)
    first_down_bra = newindex!(store)
    first_up_bra = newindex!(store)
    previdx_down_ket = first_down_ket
    previdx_down_bra = first_down_bra
    for i in 1:row
        phyidx = newindex!(store)
        left_ket = newindex!(store)
        right_ket = newindex!(store)
        left_bra = newindex!(store)
        right_bra = newindex!(store)
        next_up_ket = i ==1 ? first_up_ket : previdx_down_ket
        next_up_bra = i ==1 ? first_up_bra : previdx_down_bra
        next_down_ket = i == 1 ? first_down_ket : newindex!(store)
        next_down_bra = i == 1 ? first_down_bra : newindex!(store)
        push!(index_ket, [phyidx, next_down_ket, right_ket, next_up_ket, left_ket])
        push!(index_bra, [phyidx, next_down_bra, right_bra, next_up_bra, left_bra])
        previdx_down_ket = next_down_ket
        previdx_down_bra = next_down_bra
    end

    append!(index_output, index_ket[row][2])
    append!(index_output, [index_ket[i][3] for i in 1:row])
    append!(index_output, index_bra[row][2])
    append!(index_output, [index_bra[i][3] for i in 1:row])

    append!(index_output, index_ket[1][4])
    append!(index_output, [index_ket[i][end] for i in 1:row])
    append!(index_output, index_bra[1][4])
    append!(index_output, [index_bra[i][end] for i in 1:row])
    index=[index_ket...,index_bra...]

    size_dict = OMEinsum.get_size_dict(index, [tensor_ket...,tensor_bra...])
    code = optimize_code(DynamicEinCode(index, index_output), size_dict, optimizer)
    return code, code(tensor_ket...,tensor_bra...)
end

function exact_energy_PEPS(d::Int,D::Int,g::Float64,row::Int)
    mps = InfiniteMPS([ComplexSpace(d) for _ in 1:row], [ComplexSpace(D) for _ in 1:row])
    H0 = transverse_field_ising(InfiniteCylinder(row); g=g)
    psi,_= find_groundstate(mps, H0, VUMPS())
    E = real(expectation_value(psi,H0))/row
    return E
end

function cost_X_circ(rho, row, gate; niters=50)
    rho2 = join(rho, density_matrix(zero_state(1)))
    rho2 = Yao.apply!(rho2, put(2+row,(1, 2, 3)=>gate))   
    Yao.apply!(rho2, put(2+row, 1=>H)) 
    return 1-2*mean(measure(rho2, 1; nshots=1000000))
end

function cost_ZZ_circ(rho, row, gate; niters=50)
    rho2 = join(rho, density_matrix(zero_state(1)))
    rho2 = Yao.apply!(rho2, put(2+row,(1, 2, 3)=>gate))   
    Z1 = measure(rho2, 1; nshots=1000000)
    Z1 = 1-2*mean(Z1)
    # Trace out the measured qubit to reduce register size
    rho2 = partial_tr(rho2, 1)
    rho2 = join(rho2, density_matrix(zero_state(1)))
    Yao.apply!(rho2, put(2+row, (1,2,4)=>gate))
    Z2 = measure(rho2, 1; nshots=1000000)
    Z2 = 1-2*mean(Z2)
    
    return Z1*Z2
end

function cost_X(rho, row, gate)
    nqubits = Int(log2(size(rho.state, 1)))
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho.state, shape...)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    AX = ein"iabcd,ij -> jabcd"(A, Matrix(X))
    tensor_ket = [AX, A, A]
    tensor_bra = [conj(A), conj(A), conj(A)]
    _, list= contract_Elist(tensor_ket, tensor_bra, row)
    result = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list)

    return result[]
end

function cost_ZZ(rho, row, gate)
    nqubits = Int(log2(size(rho.state, 1)))
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho.state, shape...)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    AZ = ein"iabcd,ij -> jabcd"(A, Matrix(Z))
    tensor_ket = [AZ, AZ, A]
    tensor_bra = [conj(A), conj(A), conj(A)]
    _, list1= contract_Elist(tensor_ket, tensor_bra, row)
    result1 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list1)
    _, list2= contract_Elist([A, AZ, AZ], tensor_bra, row)
    result2 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list2)
    _, list3 = contract_Elist([AZ, A, AZ], tensor_bra, row)
    result3 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list3)
    @show result1[], result2[]
    return 2*(result1[]+result2[]+result3[])/3
end

function train_energy_circ(params, J::Float64, g::Float64, p::Int, row::Int; maxiter=5000, step_size=0.1)
    X_history = Float64[]
    final_A = Matrix(I, 8, 8)
    final_params = []

    function objective(x)
        A_matrix = Matrix(I, 8, 8)
        for r in 1:p
            #gate = kron(Ry(x[3*r-2]), Ry(x[3*r-1]), Ry(x[3*r]))
            gate = kron(Rx(x[6*r-5])*Rz(x[6*r-4]), Rx(x[6*r-3])*Rz(x[6*r-2]), Rx(x[6*r-1])*Rz(x[6*r]))
            cnot_12 = cnot(3,2,1)
            cnot_23 = cnot(3,3,2)
            cnot_31 = cnot(3,1,3)
            A_matrix *= Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
        end
        @assert A_matrix * A_matrix' ≈ I atol=1e-5
        @assert A_matrix' * A_matrix ≈ I atol=1e-5
        gate = matblock(A_matrix)
       
        rho = iterate_channel_PEPS(gate, row)
        #energy = -g*cost_X_circ(rho, row, gate) - 2*J*cost_ZZ_circ(rho, row, gate)  # by measurement
        energy = -g*cost_X(rho, row, gate) - J*cost_ZZ(rho, row, gate)  # by contraction
        push!(X_history, real(energy))
        @info "Iter $(length(X_history)), cost: $energy"
        final_A = A_matrix
        final_params = x

        return real(energy)
    end
    @info "Number of parameters is $(length(params))"
  
    optimizer = Optim.NelderMead(; 
        parameters = Optim.AdaptiveParameters(),
        initial_simplex = Optim.AffineSimplexer()
    )
    
    result = Optim.optimize(objective, params, optimizer, Optim.Options(
        iterations=maxiter,
        show_trace=true,
        f_reltol = 1e-6,  
      
        g_tol = 1e-6,     
        x_reltol=1e-6,
        #f_abstol=1.99,
        time_limit=3600.0 
    ))

    final_cost = Optim.minimum(result)
    # Optional: you can also return the full optimization result for more details
    # return X_history, final_A, final_params, final_cost, result
    @show final_cost
    return X_history, final_A, final_params, final_cost
end

function train_nocompile(gate, row, M::AbstractManifold, J::Float64, g::Float64; maxiter=3000)
    
    function f(M, gate)
        gate = matblock(gate)
        rho = iterate_channel_PEPS(gate, row)
        energy = -g*cost_X(rho, row, gate) - J*cost_ZZ(rho, row, gate)
        return real(energy)
    end

    @assert is_point(M, Matrix(gate))

    result = Manopt.NelderMead(
        M, 
        f,
        population=NelderMeadSimplex(M);
        stopping_criterion = StopAfterIteration(maxiter) | StopWhenPopulationConcentrated(1e-4, 1e-4), # StopWhenPopulationConcentrated(tol_f::Real=1e-8, tol_x::Real=1e-8)
        record = [:Iteration, :Cost],
        return_state = true
    )
    
    final_p = get_solver_result(result)
    final_energy = f(M, final_p)
   
    @show final_energy
    return result, final_energy, final_p
end


function all_simulation()

end

function draw()
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
    MPSKit_list = [ -1.9999999999999971,  -2.0078141005791723, -2.031275809659576, -2.0704448852547266, -2.125426272074635, -2.1963790626176034, -2.283531518020085, -2.3872064546848364,  -2.507866896802187]
    measure_list = [ 8.003535045058994, 6.352683335061295, 5.663331887759824, 5.990444892769451, 6.396219818662397, 5.0233592901626105, 5.082605585770299, 4.874217984727903, 3.4996020960549186, 3.105510917741358]
    fig = Plots.plot(xlabel="g", ylabel="gap", title="Spectral gap vs Transverse Field", 
                     ylims=(3.0, 8.5), xlims=(-0.25, 2.5), 
                     yscale=:linear, 
                     yticks=[8.5, 8.0, 7.5, 7.0, 6.5, 6, 5.5, 5, 4.5, 4.0, 3.5, 3.0],
                     xticks=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25])

    colors = [RGBA(1,0,0,0.5),RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), RGBA(0,0,1,0.5)]
    markers = [:+, :x, :star,:diamond]
    #Plots.plot!(fig, g_list, MPSKit_list, label="iPEPS",color=colors[1], marker=markers[1],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, measure_list, label="spectral gap, iter=100",color=colors[4], marker=markers[4],markersize=4,linewidth=2)

    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:topright, legendfontsize=10)
    Plots.savefig(fig, "spectral gap_vs_g.png")
    Plots.display(fig)
end

function exact_iPEPS(d::Int, D::Int, J::Float64, g::Float64; χ ::Int= 20, ctmrg_tol::Float64= 1e-10, grad_tol::Float64= 1e-4, maxiter::Int=1000)
    H = transverse_field_ising(PEPSKit.InfiniteSquare(); g)
    peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χ)), peps₀; tol=ctmrg_tol)
    peps, env, E, = fixedpoint(H, peps₀, env₀; tol=grad_tol, boundary_alg=(; tol=ctmrg_tol), 
                              optimizer_alg=(; maxiter=maxiter))
    return E
end

