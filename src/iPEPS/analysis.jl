"""
Analysis tools for iPEPS optimization results.

Provides sensitivity analysis, convergence checking, and parameter
importance quantification.
"""

"""
    check_gap_sensitivity(params::Vector{Float64}, param_idx::Int, g::Float64, 
                         row::Int, p::Int; param_range=range(0, 2π, length=50),
                         save_path=nothing)

Analyze sensitivity of spectral gap to a single parameter.

# Arguments
- `params`: Parameter vector
- `param_idx`: Index of parameter to vary
- `g`: Transverse field strength
- `row`: Number of rows
- `p`: Number of circuit layers
- `param_range`: Range of parameter values to test
- `save_path`: Path to save plot (optional)

# Returns
- `param_values`: Values tested
- `gap_values`: Spectral gaps
- `energy_values`: Energies

# Description
Varies a single parameter while keeping others fixed, measuring the
impact on spectral gap and energy. Useful for identifying critical
parameters.
"""
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
        A_matrix = build_gate_from_params(test_params, p)
        
        # Check unitarity
        if !(A_matrix * A_matrix' ≈ I)
            @warn "Gate not unitary at parameter value $param_val, skipping"
            continue
        end
        
        gate_block = matblock(A_matrix)
        
        # Compute spectral gap
        rho, gap = exact_left_eigen(gate_block, row)
        push!(gap_values, gap)
        
        # Compute energy by contraction
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

"""
    check_all_gap_sensitivity_combined(params::Vector{Float64}, g::Float64, 
                                       row::Int, p::Int; 
                                       param_range=range(0, 2π, length=50),
                                       save_path=nothing, plot_energy=true)

Analyze sensitivity of all parameters simultaneously.

# Arguments
- `params`: Parameter vector
- `g`: Transverse field strength
- `row`: Number of rows
- `p`: Number of circuit layers
- `param_range`: Range to test for each parameter
- `save_path`: Path to save combined plot
- `plot_energy`: Whether to plot energy alongside gap

# Returns
- `all_param_values`: Parameter values tested for each parameter
- `all_gap_values`: Gap values for each parameter
- `all_energy_values`: Energy values for each parameter
- `gap_sensitivities`: Standard deviation of gaps (sensitivity metric)

# Description
Performs sensitivity analysis for all parameters, creating a combined
plot and ranking parameters by their importance. Identifies which
parameters most strongly affect the spectral gap and energy.
"""
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
            A_matrix = build_gate_from_params(test_params, p)
            
            # Check unitarity
            if !(A_matrix * A_matrix' ≈ I)
                @warn "Gate not unitary at parameter value $param_val for param $param_idx, skipping"
                continue
            end
            
            gate_block = matblock(A_matrix)
            
            # Compute spectral gap
            rho, gap = exact_left_eigen(gate_block, row)
            push!(gap_values, gap)
            
            # Compute energy
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

