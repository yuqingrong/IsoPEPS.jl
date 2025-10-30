function iterate_channel_PEPS(gate, row; niters=10000)
    rho = zero_state(row+1)
    Z1_list = Float64[]
    X1_list = Float64[]
    for i in 1:niters
        for j in 1:row
            rho_p = zero_state(1)
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>gate)) 

            #if i > niters ÷ 2 
                if i > niters*3 ÷ 4  
                    Z = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(Z1_list, Z.buf)
                else
                    Yao.apply!(rho, put(2+row, 1=>H)) 
                    X = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(X1_list, X.buf)
                end
            #else
                #measure!(RemoveMeasured(), rho, 1)
            #end
        end
        #@info "eigenvalues of ρ = " eigen(Hermitian(rho.state)).values
        #@show isapprox(rho.state, rho.state') 
    end
    return rho, Z1_list, X1_list
end

function exact_left_eigen(gate, nsites)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    #E = reshape(ein"iabcd,iefgh -> abefcdgh"(A, conj(A)), 4,4,4,4)
    _, T = contract_Elist([A for _ in 1:nsites], [conj(A) for _ in 1:nsites], nsites)
    T = reshape(T, 4^(nsites+1), 4^(nsites+1))
    λ₁ = partialsort(abs.(LinearAlgebra.eigen(T).values),2;rev=true)
    #λ₁ = LinearAlgebra.eigen(T).values[end-1]
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
    nqubits = Int(log2(size(rho, 1))) # TODO: rho -> rho.state
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho, shape...)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    AX = ein"iabcd,ij -> jabcd"(A, Matrix(X))
    tensor_ket = [AX, A, A]
    tensor_bra = [conj(A), conj(A), conj(A)]
    _, list= contract_Elist(tensor_ket, tensor_bra, row)
    result = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list)

    return result[]
end

function cost_ZZ(rho, row, gate)
    nqubits = Int(log2(size(rho, 1)))
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho, shape...)
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
    return 2*(result1[]+result2[]+result3[])/3
end

function train_energy_circ(params, J::Float64, g::Float64, p::Int, row::Int; maxiter=5000)
    X_history = Float64[]
    params_history = Vector{Float64}[]
    final_A = Matrix(I, 8, 8)
    Z_list_list = Vector{Float64}[]
    X_list_list = Vector{Float64}[]
    gap_list = Float64[]
    current_params = copy(params)
    iter_count = Ref(0)
    
    function objective(x,_)
        iter_count[] += 1
        
        # Hard stop after maxiter function evaluations
        if iter_count[] > maxiter
            @warn "Reached maximum iterations ($maxiter). Stopping..."
            #save_training_data(Z_list_list, X_list_list, gap_list)
            error("Maximum iterations reached")
        end
        
        current_params .= x
        push!(params_history, copy(x))
        A_matrix = Matrix(I, 8, 8)
        for r in 1:p
            #gate = kron(Ry(x[3*r-2]), Ry(x[3*r-1]), Ry(x[3*r]))
            gate = kron(Yao.Rx(x[6*r-5])*Yao.Rz(x[6*r-4]), Yao.Rx(x[6*r-3])*Yao.Rz(x[6*r-2]), Yao.Rx(x[6*r-1])*Yao.Rz(x[6*r]))
            cnot_12 = cnot(3,2,1)
            cnot_23 = cnot(3,3,2)
            cnot_31 = cnot(3,1,3)
            A_matrix *= Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
        end
        @assert A_matrix * A_matrix' ≈ I atol=1e-5
        @assert A_matrix' * A_matrix ≈ I atol=1e-5
        gate = matblock(A_matrix)
       
        rho, Z_list, X_list = iterate_channel_PEPS(gate, row)
        _, gap = exact_left_eigen(gate, row)
        push!(gap_list, gap)
        push!(Z_list_list, Z_list)
        push!(X_list_list, X_list)
        #energy = -g*cost_X_circ(rho, row, gate) - 2*J*cost_ZZ_circ(rho, row, gate)  # by measurement
        #energy = -g*cost_X(rho, row, gate) - J*cost_ZZ(rho, row, gate)  # by contraction
        Z1_list = Z_list[1:2*row:end]
        Z2_list = Z_list[2:2*row:end]  
        Z3_list = Z_list[3:2*row:end]
        Z4_list = Z_list[4:2*row:end]
        Z5_list = Z_list[5:2*row:end]
        Z6_list = Z_list[6:2*row:end]
        energy = -g*mean(X_list[end-7500:end]) - J*mean(Z1_list .* Z2_list + Z2_list .* Z3_list + Z1_list .* Z3_list + Z1_list .* Z4_list + Z2_list .* Z5_list + Z3_list .* Z6_list)/row
        #TODO: modify the 7500
        push!(X_history, real(energy))
        @info "Iter $(length(X_history)), energy: $energy, gap: $gap"

        return real(energy)
    end
    @info "Number of parameters is $(length(params))"
  
    #==
    optimizer = Optim.NelderMead(; parameters = Optim.AdaptiveParameters(),
    initial_simplex = Optim.AffineSimplexer())
    result = Optim.optimize(objective, params, optimizer, Optim.Options(
        iterations=maxiter,
        show_trace=true,
        f_reltol = 1e-6,  
      
        g_tol = 1e-6,     
        x_reltol=1e-6,
        #f_abstol=1.99,
        time_limit=3600.0 
    ))
==#
    f = OptimizationFunction(objective)
    prob = Optimization.OptimizationProblem(f, params, lb = zeros(length(params)), ub = fill(2*pi, length(params)))
    
    local final_params, final_cost
    try
        sol = solve(prob, CMAEvolutionStrategyOpt(), maxiters=5000, reltol=1e-6, abstol=1e-6)
        final_params = sol.u
        final_cost = sol.objective
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Using parameters from iteration $maxiter"
            final_params = current_params
            final_cost = isempty(X_history) ? NaN : X_history[end]
        else
            rethrow(e)
        end
    end   
    #==
    final_cost = result.minimum
    final_params = result.minimizer==#
    for r in 1:p
        gate = kron(Yao.Rx(final_params[6*r-5])*Yao.Rz(final_params[6*r-4]), 
                    Yao.Rx(final_params[6*r-3])*Yao.Rz(final_params[6*r-2]), 
                    Yao.Rx(final_params[6*r-1])*Yao.Rz(final_params[6*r]))
        cnot_12 = cnot(3,2,1)
        cnot_23 = cnot(3,3,2)
        cnot_31 = cnot(3,1,3)
        final_A *= Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
    end
    @show final_cost
    return X_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history
end

# check convergence
function draw_X_from_file(g::Float64, n::Vector{Int}; data_dir="data", save_path=nothing, block_size=1000)
    g_str = string(g)

    filename = joinpath(data_dir, "X_list_g=$g.dat") 
    if !isfile(filename)
        @error "File not found: $filename"
        return nothing
    end
    
    @info "Reading $filename..."
    
    # Read all data lines (skip comments and empty lines)
    data_lines = []
    open(filename, "r") do file
        for line in eachline(file)
            if !startswith(strip(line), "#") && !isempty(strip(line))
                push!(data_lines, line)
            end
        end
    end
    
    total_lines = length(data_lines)
    @info "Found $total_lines data lines in file"
    
    # Create the plot
    p = Plots.plot(xlabel="Number of Blocks Used", ylabel="Block Variance", 
             title="Block Variance Analysis (g=$g, block_size=$block_size)", 
             legend=:best, yscale=:log10,ylims=(1e-4,1e-2))
    
    # Process each requested line
    for line_idx in n
        if line_idx < 1 || line_idx > total_lines
            @warn "Line $line_idx out of range (1-$total_lines), skipping"
            continue
        end
        
        # Parse the values from this line
        line = data_lines[line_idx]
        try
            values = parse.(Float64, split(line))
            
            if length(values) < 2 * block_size
                @warn "Line $line_idx has only $(length(values)) values (< $(2*block_size)), skipping"
                continue
            end
            
            # Calculate block means
            n_blocks = div(length(values), block_size)
            block_means = Float64[]
            for i in 1:n_blocks
                block_start = (i-1) * block_size + 1
                block_end = i * block_size
                block_mean = mean(values[block_start:block_end])
                push!(block_means, block_mean)
            end
            
            # Calculate cumulative block variance
            # (variance computed using first k blocks)
            block_var = Float64[]
            for k in 2:length(block_means)
                var_k = var(block_means[1:k])
                push!(block_var, var_k)
            end
            
            # Plot block variance
            Plots.plot!(p, 2:length(block_means), block_var, label="params_group $line_idx", linewidth=2, marker=:circle, markersize=3)
            @info "Plotted line $line_idx: $(n_blocks) blocks, final variance = $(block_var[end])"
            
        catch e
            @warn "Error processing line $line_idx: $e"
            continue
        end
    end
    
    # Save the figure
    if save_path !== false
        if save_path === nothing
            # Auto-generate filename
            lines_str = join(n, "_")
            save_path = "X_block_var_g=$(g_str)_lines_$(lines_str).pdf"
        end
        savefig(p, save_path)
        @info "Figure saved to: $save_path"
    end
    
    return p
end

function check_gap_sensitivity(params::Vector{Float64}, param_idx::Int, g::Float64, row::Int, p::Int;
                               param_range=range(0, 2π, length=50),
                               save_path=nothing)
    if param_idx < 1 || param_idx > length(params)
        error("param_idx must be between 1 and $(length(params))")
    end
    
    @info "Checking sensitivity of parameter $param_idx"
    @info "Parameter range: $(first(param_range)) to $(last(param_range))"
    
    param_values = collect(param_range)
    gap_values = Float64[]
    energy_values = Float64[]
    
    J = 1.0  # Default coupling
    
    for (i, param_val) in enumerate(param_values)
        # Create a copy of parameters with one parameter changed
        test_params = copy(params)
        test_params[param_idx] = param_val
        
        # Build the gate from parameters
        A_matrix = Matrix(I, 8, 8)
        for r in 1:p
            gate = kron(Yao.Rx(test_params[6*r-5])*Yao.Rz(test_params[6*r-4]), 
                        Yao.Rx(test_params[6*r-3])*Yao.Rz(test_params[6*r-2]), 
                        Yao.Rx(test_params[6*r-1])*Yao.Rz(test_params[6*r]))
            cnot_12 = cnot(3,2,1)
            cnot_23 = cnot(3,3,2)
            cnot_31 = cnot(3,1,3)
            A_matrix *= Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
        end
        
        # Check unitarity
        if !(A_matrix * A_matrix' ≈ I)
            @warn "Gate not unitary at parameter value $param_val, skipping"
            continue
        end
        
        gate_block = matblock(A_matrix)
        
        # Compute spectral gap
        rho, gap = exact_left_eigen(gate_block, row)
        push!(gap_values, gap)
        
        # Optionally compute energy
        #==
        rho, Z_list, X_list = iterate_channel_PEPS(gate_block, row; niters=10000)
        Z1_list = Z_list[1:2*row:end]
        Z2_list = Z_list[2:2*row:end]  
        Z3_list = Z_list[3:2*row:end]
        Z4_list = Z_list[4:2*row:end]
        Z5_list = Z_list[5:2*row:end]
        Z6_list = Z_list[6:2*row:end]
        
        
        energy = -g*mean(X_list[end-7500:end]) -     
                 J*mean(Z1_list[1:2*row:end] .* Z2_list[1:2*row:end] + 
                        Z2_list[1:2*row:end] .* Z3_list[1:2*row:end] + 
                        Z1_list[1:2*row:end] .* Z3_list[1:2*row:end] + 
                        Z1_list[1:2*row:end] .* Z4_list[1:2*row:end] + 
                        Z2_list[1:2*row:end] .* Z5_list[1:2*row:end] + 
                        Z3_list[1:2*row:end] .* Z6_list[1:2*row:end])/row 
        ==#
        energy = -g*cost_X(rho, row, gate_block) - J*cost_ZZ(rho, row, gate_block)
        push!(energy_values, real(energy))
        
        if i % 10 == 0
            @info "Progress: $i/$(length(param_values)) - Current gap: $gap, energy: $(real(energy))"
        end
    end
    # Plot results
    p1 = Plots.plot(param_values, gap_values, 
                    xlabel="Parameter_idx= $param_idx Value", 
                    ylabel="Spectral Gap",
                    ylims=(0,10.0),
                    title="Gap Sensitivity (g=$g, param_idx=$param_idx)",
                    linewidth=2, marker=:circle, markersize=3,
                    legend=false)
    
    p2 = Plots.plot(param_values, energy_values,
                    xlabel="Parameter_idx $param_idx Value",
                    ylabel="Energy",
                    ylims=(-3.0,7.0),
                    title="Energy (g=$g, param_idx=$param_idx)",
                    linewidth=2, marker=:circle, markersize=3,
                    legend=false)
    
    p = Plots.plot(p1, p2, layout=(2,1), size=(800, 800))
    
    # Save results if requested
    if save_path !== nothing
        Plots.savefig(p, save_path)
        @info "Figure saved to: $save_path"
        
        # Also save data to file
        data_path = replace(save_path, r"\.(png|pdf|svg)$" => ".dat")
        open(data_path, "w") do io
            println(io, "# Spectral gap sensitivity analysis")
            println(io, "# Parameter index: $param_idx")
            println(io, "# Transverse field g: $g")
            println(io, "# Columns: parameter_value gap energy")
            println(io, "#")
            for i in 1:length(param_values)
                println(io, "$(param_values[i]) $(gap_values[i]) $(energy_values[i])")
            end
        end
        @info "Data saved to: $data_path"
    end
    
    Plots.display(p)
    
    return param_values, gap_values, energy_values
end


function check_all_gap_sensitivity_combined(params::Vector{Float64}, g::Float64, row::Int, p::Int;
                                            param_range=range(0, 2π, length=50),
                                            save_path=nothing,
                                            plot_energy=true)

    n_params = length(params)
    @info "Computing gap sensitivity for all $n_params parameters"
    @info "Parameter range: $(first(param_range)) to $(last(param_range)) with $(length(param_range)) points"
    
    # Storage for results
    all_param_values = Vector{Vector{Float64}}()
    all_gap_values = Vector{Vector{Float64}}()
    all_energy_values = Vector{Vector{Float64}}()
    gap_sensitivities = Float64[]
    
    J = 1.0  # Default coupling
    
    # Compute sensitivity for each parameter
    for param_idx in 1:n_params
        @info "\nProcessing parameter $param_idx/$n_params"
        
        param_values = collect(param_range)
        gap_values = Float64[]
        energy_values = Float64[]
        
        for (i, param_val) in enumerate(param_values)
            # Create a copy of parameters with one parameter changed
            test_params = copy(params)
            test_params[param_idx] = param_val
            
            # Build the gate from parameters
            A_matrix = Matrix(I, 8, 8)
            for r in 1:p
                gate = kron(Yao.Rx(test_params[6*r-5])*Yao.Rz(test_params[6*r-4]), 
                            Yao.Rx(test_params[6*r-3])*Yao.Rz(test_params[6*r-2]), 
                            Yao.Rx(test_params[6*r-1])*Yao.Rz(test_params[6*r]))
                cnot_12 = cnot(3,2,1)
                cnot_23 = cnot(3,3,2)
                cnot_31 = cnot(3,1,3)
                A_matrix *= Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
            end
            
            # Check unitarity
            if !(A_matrix * A_matrix' ≈ I)
                @warn "Gate not unitary at parameter value $param_val for param $param_idx, skipping"
                continue
            end
            
            gate_block = matblock(A_matrix)
            
            # Compute spectral gap
            rho, gap = exact_left_eigen(gate_block, row)
            push!(gap_values, gap)
            
            # Optionally compute energy
            #==
            if plot_energy
                rho, Z_list, X_list = iterate_channel_PEPS(gate_block, row; niters=10000)
                Z1_list = Z_list[1:2*row:end]
                Z2_list = Z_list[2:2*row:end]  
                Z3_list = Z_list[3:2*row:end]
                Z4_list = Z_list[4:2*row:end]
                Z5_list = Z_list[5:2*row:end]
                Z6_list = Z_list[6:2*row:end]
                
                energy = -g*mean(X_list[end-7500:end]) -     
                J*mean(Z1_list[1:2*row:end] .* Z2_list[1:2*row:end] + 
                       Z2_list[1:2*row:end] .* Z3_list[1:2*row:end] + 
                       Z1_list[1:2*row:end] .* Z3_list[1:2*row:end] + 
                       Z1_list[1:2*row:end] .* Z4_list[1:2*row:end] + 
                       Z2_list[1:2*row:end] .* Z5_list[1:2*row:end] + 
                       Z3_list[1:2*row:end] .* Z6_list[1:2*row:end])/row 
                push!(energy_values, real(energy))
            end
            ==#
            energy = -g*cost_X(rho, row, gate_block) - J*cost_ZZ(rho, row, gate_block)
            push!(energy_values, real(energy))
            if i % 10 == 0
                @info "  Parameter $param_idx: $i/$(length(param_values)) points completed"
            end
        end
        
        # Store results
        push!(all_param_values, param_values)
        push!(all_gap_values, gap_values)
        push!(all_energy_values, energy_values)
        
        # Calculate sensitivity metric
        gap_std = std(gap_values)
        push!(gap_sensitivities, gap_std)
        
        @info "Parameter $param_idx: Gap range [$(minimum(gap_values)), $(maximum(gap_values))], std = $gap_std"
    end
    
    # Create combined plot for gaps
    @info "\nCreating combined plot..."
    
    # Generate colors for each parameter
    colors = palette(:tab20, n_params)
    
    p_gap = Plots.plot(xlabel="Parameter Value (radians)", 
                       ylabel="Spectral Gap",
                       title="Spectral Gap Sensitivity for All Parameters (g=$g)",
                       legend=:outertopright,
                       size=(1200, 600),
                       legendfontsize=7,
                       legendcolumns=2)
    
    for param_idx in 1:n_params
        Plots.plot!(p_gap, all_param_values[param_idx], all_gap_values[param_idx], 
                    label="Param $param_idx", 
                    linewidth=1.5,
                    alpha=0.7,
                    color=colors[param_idx])
    end
    
    # Optionally create energy plot
    if plot_energy
        p_energy = Plots.plot(xlabel="Parameter Value (radians)", 
                             ylabel="Energy",
                             title="Energy for All Parameters (g=$g)",
                             legend=:outertopright,
                             size=(1200, 600),
                             legendfontsize=7,
                             legendcolumns=2)
        
        for param_idx in 1:n_params
            Plots.plot!(p_energy, all_param_values[param_idx], all_energy_values[param_idx], 
                        label="Param $param_idx", 
                        linewidth=1.5,
                        alpha=0.7,
                        color=colors[param_idx])
        end
        
        combined_plot = Plots.plot(p_gap, p_energy, layout=(2,1), size=(1200, 1000))
    else
        combined_plot = p_gap
    end
    
    # Save if requested
    if save_path !== nothing
        Plots.savefig(combined_plot, save_path)
        @info "Combined plot saved to: $save_path"
        
        # Save data to file
        data_path = replace(save_path, r"\.(png|pdf|svg)$" => ".dat")
        open(data_path, "w") do io
            println(io, "# Combined spectral gap sensitivity analysis")
            println(io, "# Transverse field g: $g")
            println(io, "# Number of parameters: $n_params")
            println(io, "# Parameter range: $(first(param_range)) to $(last(param_range))")
            println(io, "#")
            println(io, "# Sensitivity summary (standard deviation of gaps):")
            for i in 1:n_params
                println(io, "# Parameter $i: std = $(gap_sensitivities[i])")
            end
            println(io, "#")
            println(io, "# Data format: Each block contains data for one parameter")
            println(io, "# Columns: parameter_value gap $(plot_energy ? "energy" : "")")
            println(io, "#")
            
            for param_idx in 1:n_params
                println(io, "\n# Parameter $param_idx")
                for i in 1:length(all_param_values[param_idx])
                    if plot_energy
                        println(io, "$(all_param_values[param_idx][i]) $(all_gap_values[param_idx][i]) $(all_energy_values[param_idx][i])")
                    else
                        println(io, "$(all_param_values[param_idx][i]) $(all_gap_values[param_idx][i])")
                    end
                end
            end
        end
        @info "Data saved to: $data_path"
    end
    
    Plots.display(combined_plot)
    
    # Print summary
    @info "\n" * "="^60
    @info "SENSITIVITY ANALYSIS SUMMARY"
    @info "="^60
    sorted_indices = sortperm(gap_sensitivities, rev=true)
    @info "\nMost sensitive parameters (top 5):"
    for i in 1:min(5, length(sorted_indices))
        idx = sorted_indices[i]
        @info "  Parameter $idx: std = $(gap_sensitivities[idx])"
    end
    @info "\nLeast sensitive parameters (bottom 5):"
    for i in max(1, length(sorted_indices)-4):length(sorted_indices)
        idx = sorted_indices[i]
        @info "  Parameter $idx: std = $(gap_sensitivities[idx])"
    end
    
    return all_param_values, all_gap_values, all_energy_values, gap_sensitivities
end

function draw_gap()
    
    # Define g values to search for
    g_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    # Store results
    valid_g_values = Float64[]
    average_gaps = Float64[]
    
    # Process each g value
    for g in g_values
        # Try both formats: "g=X.X" and "g=X.XX"
        filenames = ["data/gap_list_g=$(g).dat", "data/gap_list_g=$(round(g, digits=2)).dat"]
        
        gap_data = nothing
        for filename in filenames
            if isfile(filename)
                try
                    # Read the file
                    lines = readlines(filename)
                    
                    # Skip comment lines (starting with #)
                    data_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), lines)
                    
                    if !isempty(data_lines)
                        # Parse the gap values (assuming one value per line or space-separated)
                        gap_values_list = Float64[]
                        for line in data_lines
                            # Try to parse as a single number or take the first number if space-separated
                            parts = split(strip(line))
                            if !isempty(parts)
                                val = parse(Float64, parts[1])
                                push!(gap_values_list, val)
                            end
                        end
                        
                        if !isempty(gap_values_list)
                            gap_data = gap_values_list
                            break
                        end
                    end
                catch e
                    @warn "Error reading file $filename: $e"
                end
            end
        end
        
        # Calculate average of last 50 elements if data was found
        if gap_data !== nothing && length(gap_data) >= 50
            last_50 = gap_data[end-9:end]
            avg_gap = mean(last_50)
            push!(valid_g_values, g)
            push!(average_gaps, avg_gap)
            @info "g = $g: average gap (last 50) = $avg_gap"
        elseif gap_data !== nothing
            # If less than 50 elements, use all available data
            avg_gap = mean(gap_data)
            push!(valid_g_values, g)
            push!(average_gaps, avg_gap)
            @info "g = $g: average gap (all $(length(gap_data)) points) = $avg_gap"
        else
            @warn "No data found for g = $g"
        end
    end
    
    # Create the plot
    if !isempty(valid_g_values)
        p = Plots.plot(
            valid_g_values,
            average_gaps,
            xlabel="g",
            ylabel="Spectral Gap",
            title="Spectral Gap vs g (averaged over last 50 iterations)",
            legend=false,
            marker=:circle,
            markersize=6,
            linewidth=2,
            size=(800, 600)
        )
        
        # Save the plot
        savefig(p, "spectral gap_vs_g.pdf")
        @info "Plot saved as 'spectral gap_vs_g.png'"
        
        return p
    else
        @error "No valid data found to plot"
        return nothing
    end
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


function save_training_data(Z_list_list, X_list_list, gap_list; 
                           z_file="data/z_list_list_data.dat",
                           x_file="data/x_list_list_data.dat", 
                           gap_file="data/gap_list_data.dat")
    # Save Z_list_list
    open(z_file, "w") do io
        println(io, "# Z_list_list data from iPEPS optimization")
        println(io, "# Number of iterations: $(length(Z_list_list))")
        if !isempty(Z_list_list)
            println(io, "# Length of each Z_list: $(length(Z_list_list[1]))")
        end
        println(io, "# Format: Each line contains one Z_list (space-separated values)")
        println(io, "#")
        for z_list in Z_list_list
            println(io, join(z_list, " "))
        end
    end
    @info "Z_list_list saved to $z_file"
    
    # Save X_list_list
    open(x_file, "w") do io
        println(io, "# X_list_list data from iPEPS optimization")
        println(io, "# Number of iterations: $(length(X_list_list))")
        if !isempty(X_list_list)
            println(io, "# Length of each X_list: $(length(X_list_list[1]))")
        end
        println(io, "# Format: Each line contains one X_list (space-separated values)")
        println(io, "#")
        for x_list in X_list_list
            println(io, join(x_list, " "))
        end
    end
    @info "X_list_list saved to $x_file"
    
    # Save gap_list
    open(gap_file, "w") do io
        println(io, "# Spectral gap data from iPEPS optimization")
        println(io, "# Number of iterations: $(length(gap_list))")
        println(io, "# Format: Space-separated values")
        println(io, "#")
        println(io, join(gap_list, " "))
    end
    @info "gap_list saved to $gap_file"
end

function draw()
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    MPSKit_list = [ -1.9999999999999971,  -2.0078141005791723, -2.031275809659576, -2.0704448852547266, -2.125426272074635, -2.1963790626176034, -2.283531518020085, -2.3872064546848364,  -2.507866896802187]
    PEPSKitlist = [-1.999999610358945, -2.007814504995826, -2.0312864807194675, -2.0705176991971634, -2.1256518211812336, -2.1969439738505, -2.2846818634108392, -2.389277196571146, -2.511299175269]
    nocompile_list = [-1.9999999831857058, -2.0075969924758588, -2.031231043061209, -2.073959788377221, -2.1257631690404013,-2.1954253120312677,-2.2854081065855065,-2.388889424534069, -2.5077622489855362 ]
    contract_list = [-1.9999998917735167, -2.0078191059955737, -2.0312516166482886, -2.0703162733046936, -2.1268495602624498, -2.197127459624827, -2.287280846863297, -2.3905645948990832,  -2.505084442515908]
    measure_list = [ -1.999536026912,  -2.00675968,  -2.030210653512, -2.0682195200000004, -2.1223441800000002, -2.19005, -2.2829,  -2.3856532799999997,  -2.5062471200000003]
    fig = Plots.plot(xlabel="g", ylabel="energy density", title="energy density vs Transverse Field", 
                     ylims=(-2.6,-1.9), xlims=(-0.25, 2.5), 
                     yscale=:linear, 
                     yticks=[-3.0, -2.5, -2.0, -1.5],
                     xticks=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25])

    colors = [RGBA(1,0,0,0.5),RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), RGBA(0,0,1,0.5),RGBA(0,0,0,0.5)]
    markers = [:+, :x, :star,:diamond,:circle]
    Plots.plot!(fig, g_list, MPSKit_list, label="MPSKit",color=colors[1], marker=markers[1],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, PEPSKitlist, label="PEPSKit",color=colors[2], marker=markers[2],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, contract_list, label="contract_directly",color=colors[3], marker=markers[3],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, measure_list, label="measure",color=colors[4], marker=markers[4],markersize=4,linewidth=2)
    Plots.plot!(fig, g_list, nocompile_list, label="nocompile",color=colors[5], marker=markers[5],markersize=4,linewidth=2)
    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:topright, legendfontsize=10)
    Plots.savefig(fig, "energy density_vs_g.png")
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

