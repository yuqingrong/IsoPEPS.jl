function iterate_channel_PEPS(gate, niters, row)
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
    _, T = contract_Elist(A, nsites)
    T = reshape(T, 4^(nsites+1), 4^(nsites+1))
    λ₁ = partialsort(abs.(LinearAlgebra.eigen(T).values),2;rev=true)
    @show LinearAlgebra.eigen(T).values
    @show λ₁
    gap = -log(abs(λ₁))
    @assert LinearAlgebra.eigen(T).values[end] ≈ 1.
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^(nsites+1))), Int(sqrt(4^(nsites+1))))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    @show isapprox(rho, rho') 
    return rho, gap
end

# contract list of A and A^dagger to form transfer matrix  
function contract_Elist(A, row; optimizer=GreedyMethod())
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
    tensors_ket = [A for _ in 1:row]
    tensors_bra = [conj(A) for _ in 1:row]
    size_dict = OMEinsum.get_size_dict(index, [tensors_ket...,tensors_bra...])
    code = optimize_code(DynamicEinCode(index, index_output), size_dict, optimizer)
    return code, code(tensors_ket...,tensors_bra...)
end

function exact_energy_PEPS(d::Int,D::Int,g::Float64,row::Int)
    # Create MPS with 3-site unit cell to match InfiniteStrip(3)
    mps = InfiniteMPS([ComplexSpace(d) for _ in 1:row], [ComplexSpace(D) for _ in 1:row])
    H0 = transverse_field_ising(InfiniteCylinder(row); g=g)
    psi,_= find_groundstate(mps, H0, VUMPS())
    E = real(expectation_value(psi,H0))/row
    return E
end

function cost_X_circ(gate, row; niters=50)
    rho = iterate_channel_PEPS(gate, niters, row)
    rho2 = join(rho, density_matrix(zero_state(1)))
    rho2 = Yao.apply!(rho2, put(2+row,(1, 2, 3)=>gate))   
    Yao.apply!(rho2, put(2+row, 1=>H)) 
    return 1-2*mean(measure(rho2, 1; nshots=1000000))
end

function cost_ZZ_circ(gate, row; niters=50)
    rho = iterate_channel_PEPS(gate, niters, row)
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

function cost_X(gate, row; niters=50)
    rho = iterate_channel_PEPS(gate, niters, row)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    #@assert 
    return ein
end

function cost_ZZ(gate, row; niters=50)
    rho = iterate_channel_PEPS(gate, niters, row)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    #@assert 
    return ein
end

function train_energy_circ(params, J::Float64, g::Float64, p::Int, row::Int; maxiter=1000)
    X_history = Float64[]
    final_A = Matrix(I, 8, 8)
    final_params = []

    function objective(x)
        A_matrix = Matrix(I, 8, 8)
        for r in 1:p
            gate = kron(Ry(x[3*r-2]), Ry(x[3*r-1]), Ry(x[3*r]))
            #gate = kron(Rx(x[6*r-5])*Rz(x[6*r-4]), Rx(x[6*r-3])*Rz(x[6*r-2]), Rx(x[6*r-1])*Rz(x[6*r]))
            cnot_12 = cnot(3,2,1)
            cnot_23 = cnot(3,3,2)
            cnot_31 = cnot(3,1,3)
            A_matrix *= Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
        end
        @assert A_matrix * A_matrix' ≈ I atol=1e-5
        @assert A_matrix' * A_matrix ≈ I atol=1e-5

        energy = -g*cost_X_circ(matblock(A_matrix), row) - 2*J*cost_ZZ_circ(matblock(A_matrix), row)  # by measurement
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
        f_reltol = 1e-4,  
      
        g_tol = 1e-4,     
        x_reltol=1e-4,
        #f_abstol=1.99,
        time_limit=3600.0 
    ))

    final_cost = Optim.minimum(result)
    # Optional: you can also return the full optimization result for more details
    # return X_history, final_A, final_params, final_cost, result
    @show final_cost
    return X_history, final_A, final_params, final_cost
end




function all_simulation()

end

function draw()
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    MPSKit_list = [ -1.9999999999999971,  -2.0078141005791723, -2.031275809659576, -2.0704448852547266, -2.125426272074635, -2.1963790626176034, -2.283531518020085, -2.3872064546848364,  -2.507866896802187]
    measure_list = [-1.99920008,  -2.00675968, -2.02756928, -2.0682195200000004,  -2.1175047200000003, -2.19005, -2.2829,  -2.3856532799999997,  -2.5062471200000003]
    fig = Plots.plot(xlabel="g", ylabel="E", title="Energy vs Transverse Field", 
                     ylims=(-2.6, -1.9), xlims=(-0.25, 2), 
                     yscale=:linear, 
                     yticks=[-2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9],
                     xticks=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])

    colors = [RGBA(1,0,0,0.5),RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), RGBA(0,0,1,0.5)]
    markers = [:+, :x, :star,:diamond]
    Plots.plot!(fig, g_list, MPSKit_list, label="iPEPS",color=colors[1], marker=markers[1],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, measure_list, label="measure, repeat=6",color=colors[4], marker=markers[4],markersize=4,linewidth=2)

    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:bottomright, legendfontsize=10)
    Plots.savefig(fig, "energy_vs_g.png")
    Plots.display(fig)
end

function expect(gate, nsites::Int, g::Float64)
    l = exact_left_eigen(gate, nsites)
    expect_X = ..
    expect_ZZx = ..
    expect_ZZy = ..
end

