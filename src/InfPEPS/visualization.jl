function _save_training_data(g::Float64, energy_history, params_history, Z_list_list, X_list_list, gap_list; data_dir="data", measure_first=measure_first)
    if !isdir(data_dir)
        mkdir(data_dir)
    end
    # Save energy history
    open(joinpath(data_dir, "$(measure_first)_first_energy_history_g=$(g)_.dat"), "w") do io
        for energy in energy_history
            println(io, energy)
        end
    end
    # Save params history (each row is one parameter set)
    open(joinpath(data_dir, "$(measure_first)_first_params_history_g=$(g).dat"), "w") do io
        for params in params_history
            println(io, join(params, " "))
        end
    end
    # Save Z_list_list (each row is one Z_list)
    open(joinpath(data_dir, "$(measure_first)_first_Z_list_list_g=$(g).dat"), "w") do io
        for Z_list in Z_list_list
            println(io, join(Z_list, " "))
        end
    end
    # Save X_list_list (each row is one X_list)
    open(joinpath(data_dir, "$(measure_first)_first_X_list_list_g=$(g).dat"), "w") do io
        for X_list in X_list_list
            println(io, join(X_list, " "))
        end
    end
    # Save gap list
    open(joinpath(data_dir, "$(measure_first)_first_gap_list_g=$(g).dat"), "w") do io
        for gap in gap_list
            println(io, gap)
        end
    end
    @info "Training data saved to $(data_dir)/ with g=$(g)"
end


function dynamics_observables(g::Float64; data_dir="data", measure_first=:X)
    # Construct filename
    filename = joinpath(data_dir, "$(measure_first)_first_$(measure_first)_list_list_g=$(g).dat")
    
    if !isfile(filename)
        @error "File not found: $filename"
        return nothing
    end
    @info "Reading $filename..."

    data_lines = []
    open(filename, "r") do file
        for line in eachline(file)
            if !startswith(strip(line), "#") && !isempty(strip(line))
                push!(data_lines, line)
            end
        end
    end
    
    total_lines = length(data_lines)
    if total_lines == 0
        @error "No data lines found in $filename"
        return nothing
    end
    
    @info "Found $total_lines data lines, using the last line"
    
    # Get the last line
    last_line = data_lines[end]
    
    # Parse the values from the last line
    try
        values = parse.(Float64, split(last_line))
        n_values = length(values)
        @info "Last line contains $n_values values"
        
        # Calculate cumulative means
        cumulative_means = Float64[]
        for i in 1:n_values
            cumul_mean = mean(values[1:i])
            push!(cumulative_means, cumul_mean)
        end
        @info "Calculated $(length(cumulative_means)) cumulative means"
    
        p = Plots.plot(
            1:n_values,
            cumulative_means,
            xlabel="iteration time",
            ylabel="⟨X⟩",
            title="Dynamics of X (g=$(g))",
            legend=false,
            linewidth=2,
            size=(800, 600)
        )
    
        save_path = "image/dynamics_$(measure_first)_g=$(g).pdf"
        savefig(p, save_path)
        @info "Figure saved to: $save_path"
        
        return p
        
    catch e
        @error "Error processing last line: $e"
        return nothing
    end
end

function block_variance(g::Float64, n::Vector{Int}; data_dir="data", save_path=nothing, block_size=1000)
    g_str = string(g)

    filename = joinpath(data_dir, "X_list_list_g=$g.dat") 
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
             legend=:best, yscale=:log10, ylims=(1e-4,1e-2))
    
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
            Plots.plot!(p, 2:length(block_means), block_var, 
                       label="params_group $line_idx", linewidth=2, 
                       marker=:circle, markersize=3)
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

"""
    draw_gap()

Plot spectral gap as a function of transverse field strength.

# Returns
Plot object (or nothing if data not found)

# Description
Reads gap data from multiple files (one per g value) and creates a
summary plot showing how the spectral gap varies with the transverse
field strength. Averages over the last iterations to reduce noise.
"""
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
        
        # Calculate average of last elements if data was found
        if gap_data !== nothing && length(gap_data) >= 50
            last_10 = gap_data[end-9:end]
            avg_gap = mean(last_10)
            push!(valid_g_values, g)
            push!(average_gaps, avg_gap)
            @info "g = $g: average gap (last 10) = $avg_gap"
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
            title="Spectral Gap vs g (averaged over last iterations)",
            legend=false,
            marker=:circle,
            markersize=6,
            linewidth=2,
            size=(800, 600)
        )
        
        # Save the plot
        savefig(p, "spectral gap_vs_g.pdf")
        @info "Plot saved as 'spectral gap_vs_g.pdf'"
        
        return p
    else
        @error "No valid data found to plot"
        return nothing
    end
end

"""
    draw()

Create benchmark comparison plot of energy density vs transverse field.

# Returns
Plot object

# Description
Compares different computational methods (MPSKit, PEPSKit, various
contraction methods) for computing ground state energy of the transverse
field Ising model. Uses hardcoded benchmark data.
"""
function draw()
    g_list = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    MPSKit_list = [ -1.9999999999999971,  -2.0078141005791723, -2.031275809659576, 
                    -2.0704448852547266, -2.125426272074635, -2.1963790626176034, 
                    -2.283531518020085, -2.3872064546848364,  -2.507866896802187]
    PEPSKitlist = [-1.999999610358945, -2.007814504995826, -2.0312864807194675, 
                   -2.0705176991971634, -2.1256518211812336, -2.1969439738505, 
                   -2.2846818634108392, -2.389277196571146, -2.511299175269]
    nocompile_list = [-1.9999999831857058, -2.0075969924758588, -2.031231043061209, 
                      -2.073959788377221, -2.1257631690404013,-2.1954253120312677,
                      -2.2854081065855065,-2.388889424534069, -2.5077622489855362]
    contract_list = [-1.9999998917735167, -2.0078191059955737, -2.0312516166482886, 
                     -2.0703162733046936, -2.1268495602624498, -2.197127459624827, 
                     -2.287280846863297, -2.3905645948990832,  -2.505084442515908]
    measure_list = [ -1.999536026912,  -2.00675968,  -2.030210653512, -2.0682195200000004, 
                     -2.1223441800000002, -2.19005, -2.2829,  -2.3856532799999997,  
                     -2.5062471200000003]
    
    fig = Plots.plot(xlabel="g", ylabel="energy density", 
                     title="energy density vs Transverse Field", 
                     ylims=(-2.6,-1.9), xlims=(-0.25, 2.5), 
                     yscale=:linear, 
                     yticks=[-3.0, -2.5, -2.0, -1.5],
                     xticks=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25])

    colors = [RGBA(1,0,0,0.5), RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), 
              RGBA(0,0,1,0.5), RGBA(0,0,0,0.5)]
    markers = [:+, :x, :star, :diamond, :circle]
    
    Plots.plot!(fig, g_list, MPSKit_list, label="MPSKit",
               color=colors[1], marker=markers[1], markersize=4, linewidth=2)
    Plots.plot!(fig, g_list, PEPSKitlist, label="PEPSKit",
               color=colors[2], marker=markers[2], markersize=4, linewidth=2)
    Plots.plot!(fig, g_list, contract_list, label="contract_directly",
               color=colors[3], marker=markers[3], markersize=4, linewidth=2)
    Plots.plot!(fig, g_list, measure_list, label="measure",
               color=colors[4], marker=markers[4], markersize=4, linewidth=2)
    Plots.plot!(fig, g_list, nocompile_list, label="nocompile",
               color=colors[5], marker=markers[5], markersize=4, linewidth=2)
    
    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:topright, legendfontsize=10)
    Plots.savefig(fig, "energy density_vs_g.png")
    Plots.display(fig)
    
    return fig
end

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

