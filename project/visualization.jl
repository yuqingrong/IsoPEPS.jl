function _save_training_data(g::Float64, p::Int, row::Int, nqubits::Int, energy_history, params_history, Z_list_list, X_list_list, final_gap, final_eigenvalues, final_params, final_cost; data_dir="data", measure_first=:X)
    if !isdir(data_dir)
        mkdir(data_dir)
    end
    # Save energy history
    open(joinpath(data_dir, "compile_energy_history_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g)_.dat"), "w") do io
        for energy in energy_history
            println(io, energy)
        end
    end
    # Save params history (each row is one parameter set)
    open(joinpath(data_dir, "compile_params_history_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat"), "w") do io
        for params in params_history
            println(io, join(params, " "))
        end
    end
    # Save Z_list_list (each row is one Z_list)
    open(joinpath(data_dir, "compile_Z_list_list_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat"), "w") do io
        for Z_list in Z_list_list
            println(io, join(Z_list, " "))
        end
    end

    # Save X_list_list (each row is one X_list)
    open(joinpath(data_dir, "compile_X_list_list_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat"), "w") do io
        for X_list in X_list_list
            println(io, join(X_list, " "))
        end
    end
    # Save gap list
    open(joinpath(data_dir, "compile_final_gap_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat"), "w") do io
        for gap in final_gap
            println(io, gap)
        end
    end
    # Save eigenvalues list
    open(joinpath(data_dir, "compile_final_eigenvalues_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat"), "w") do io
        for eigenvalues in final_eigenvalues
            println(io, join(eigenvalues, " "))
        end
    end
    
    open(joinpath(data_dir, "compile_final_params_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat"), "w") do io
        for params in final_params
            println(io, join(params, " "))
        end
    end

    open(joinpath(data_dir, "compile_final_cost_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat"), "w") do io
        for cost in final_cost
            println(io, cost)
        end
    end
    @info "Training data saved to $(data_dir)/ with g=$(g)"
end

function _save_training_data_exact(g::Float64, row::Int, energy_history, X_list, ZZ_list1, ZZ_list2, gap_list, eigenvalues_list, final_p; data_dir="data")
    if !isdir(data_dir)
        mkdir(data_dir)
    end
    # Save energy history
    open(joinpath(data_dir, "exact_energy_history_row=$(row)_g=$(g)_.dat"), "w") do io
        for energy in energy_history
            println(io, energy)
        end
    end

    # Save Z_list_list (each row is one Z_list)
    open(joinpath(data_dir, "exact_ZZ_list1_row=$(row)_g=$(g).dat"), "w") do io
        for Z in ZZ_list1
            println(io, Z)
        end
    end

    open(joinpath(data_dir, "exact_ZZ_list2_row=$(row)_g=$(g).dat"), "w") do io
        for Z in ZZ_list2
            println(io, Z)
        end
    end
    # Save X_list_list (each row is one X_list)
    open(joinpath(data_dir, "exact_X_list_row=$(row)_g=$(g).dat"), "w") do io
        for X in X_list
            println(io, X)
        end
    end
    # Save gap list
    open(joinpath(data_dir, "exact_gap_list_row=$(row)_g=$(g).dat"), "w") do io
        for gap in gap_list
            println(io, gap)
        end
    end

    open(joinpath(data_dir, "exact_eigenvalues_list_row=$(row)_g=$(g).dat"), "w") do io
        for eigenvalues in eigenvalues_list
            println(io, join(eigenvalues, " "))
        end
    end
    open(joinpath(data_dir, "exact_final_p_row=$(row)_g=$(g).dat"), "w") do io
        for p in final_p
            println(io, join(p, " "))
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
            ylabel="⟨$(measure_first)⟩",
            title="Dynamics of $(measure_first) (g=$(g))",
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

function dynamics_observables_all(g_values::Vector{Float64}; data_dir="data", measure_first=:X)
    
    # Initialize plot
    p = Plots.plot(
        xlabel="iteration time",
        ylabel="⟨$(measure_first)⟩",
        title="Dynamics of $(measure_first) for different g values",
        legend=:best,
        linewidth=2,
        size=(1000, 700)
    )
    
    # Track if any data was successfully plotted
    data_found = false
    
    # Process each g value
    for g in g_values
        # Construct filename
        filename = joinpath(data_dir, "$(measure_first)_first_$(measure_first)_list_list_g=$(g).dat")
        
        if !isfile(filename)
            @warn "File not found: $filename"
            continue
        end
        
        @info "Reading $filename..."
        
        # Read data lines
        data_lines = []
        try
            open(filename, "r") do file
                for line in eachline(file)
                    if !startswith(strip(line), "#") && !isempty(strip(line))
                        push!(data_lines, line)
                    end
                end
            end
        catch e
            @warn "Error reading file $filename: $e"
            continue
        end
        
        if isempty(data_lines)
            @warn "No data lines found in $filename"
            continue
        end
        
        @info "Found $(length(data_lines)) data lines for g=$g"
        
        # Get the last line
        last_line = data_lines[end]
        
        # Parse the values from the last line
        try
            values = parse.(Float64, split(last_line))
            n_values = length(values[1:5000])
            @info "Last line contains $n_values values for g=$g"
            
            # Calculate cumulative means
            cumulative_means = Float64[]
            for i in 1:n_values
                cumul_mean = mean(values[1:i])
                push!(cumulative_means, cumul_mean)
            end
            
            # Add to plot
            Plots.plot!(p, 1:n_values, cumulative_means, 
                       label="g=$(g)", 
                       linewidth=2)
            
            data_found = true
            @info "Successfully plotted data for g=$g"
            
        catch e
            @warn "Error processing data for g=$g: $e"
            continue
        end
    end
    
    if !data_found
        @error "No valid data found for any g value"
        return nothing
    end
    
    # Save the plot
    save_path = "image/dynamics_$(measure_first)_all_g.pdf"
    
    # Create image directory if it doesn't exist
    if !isdir("image")
        mkdir("image")
    end
    
    savefig(p, save_path)
    @info "Combined figure saved to: $save_path"
    
    return p
end

function eigenvalues(g_values::Vector{Float64}; data_dir="data_exact")
    # If g is a vector, plot spectrum for each g on the same figure.
    p = Plots.plot(title="Spectrum vs Eigenvalue Index", xlabel="Index", ylabel="Eigenvalue")
    for gi in g_values
        filename = joinpath(data_dir, "nocompile_eigenvalues_list_g=$(gi).dat")
        if !isfile(filename)
            @warn "File not found: $filename"
            continue
        end
        # Read non-comment, non-empty lines
        data_lines = String[]
        open(filename, "r") do file
            for line in eachline(file)
                if !startswith(strip(line), "#") && !isempty(strip(line))
                    push!(data_lines, line)
                end
            end
        end
        if isempty(data_lines)
            @warn "No valid lines found in $filename"
            continue
        end
        last_line = data_lines[end]
        eigvals = parse.(Float64, split(strip(last_line)))
        @show length(eigvals)
        plot!(1:length(eigvals), eigvals, label="spectrum g=$(gi)", linewidth=2)
    end

    # Save the plot
    save_path = "image/spectrum.pdf"
    if !isdir("image")
        mkdir("image")
    end
    savefig(p, save_path)
    @info "Spectrum figure saved to: $save_path"

    return nothing
   
end

function gap(g_values; data_dir="data")
    
    # Store results
    valid_g_values = Float64[]
    average_gaps = Float64[]
    
    # Process each g value
    for g in g_values
        filename = joinpath(data_dir, "nocompile_gap_list_g=$(g).dat")
        
        gap_data = nothing
        if isfile(filename)
            try
                # Read the file
                lines = readlines(filename)
                
                # Skip comment lines (starting with #) and empty lines
                data_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), lines)
                
                if !isempty(data_lines)
                    # Parse the gap values (assuming one value per line)
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
                    end
                end
            catch e
                @warn "Error reading file $filename: $e"
            end
        else
            @warn "File not found: $filename"
        end
        
        # Calculate average of last 50 elements if data was found
        if gap_data !== nothing && length(gap_data) >= 50
            last_50 = gap_data[end-50:end]
            avg_gap = mean(last_50)
            push!(valid_g_values, g)
            push!(average_gaps, avg_gap)
            @info "g = $g: average gap (last 50) = $avg_gap"
        elseif gap_data !== nothing && length(gap_data) > 0
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
            legend=:topright,
            marker=:circle,
            markersize=6,
            label="-ln|λ_1|, λ_1 is the 2th largest eigenvalue",
            linewidth=2,
            size=(800, 600),

        )
        
        # Save the plot
        save_path = "image/gap_vs_g.pdf"
        savefig(p, save_path)
        @info "Plot saved as '$save_path'"
        
        return p
    else
        @error "No valid data found to plot"
        return nothing
    end
end

function ACF(J::Float64, g::Float64, p::Int,row::Int, nqubits::Int; measure_first=:Z, conv_step=1000, samples=1000000, data_dir="data", save_path=nothing, max_lag=nothing, hor=true)
    filename = joinpath(data_dir, "compile_final_params_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat")
    
    if !isfile(filename)
        @error "File not found: $filename"
        return nothing
    end
    @info "Reading $filename..."
    
    # Read all data from file and form a vector
    params = []
    data_lines = []
    try
        open(filename, "r") do file
            for line in eachline(file)
                stripped = strip(line)
                if !startswith(stripped, "#") && !isempty(stripped)
                    push!(data_lines, line)
                    param_elements = parse.(Float64, split(stripped))
                    push!(params, param_elements)
                end
            end
        end
        
        if isempty(params)
            @error "No data found in $filename"
            return nothing
        end
        params = vcat(params...)
        A_matrix = build_gate_from_params(params, p, row, nqubits; share_params=true)
        rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=conv_step, samples=samples,measure_first=measure_first)
        
        # Calculate energy consistently with training (trim appropriate list by conv_step)
        if measure_first == :X
            energy = energy_measure(X_list[conv_step:end], Z_list, g, J, row)
        else
            energy = energy_measure(X_list, Z_list[conv_step:end], g, J, row)
        end
        @show energy

        if hor==true
            O = Z_list[conv_step:row:end]
        else
            O = X_list[conv_step:end]
        end
        N = length(O)
        
        # Determine maximum lag
        if max_lag === nothing
            max_lag = min(500, div(N, 2))
        else
            max_lag = min(max_lag, div(N, 2))
        end
        
        O_mean = mean(O)
        O_centered = O .- O_mean
        acf = zeros(max_lag)
        acf_err = zeros(max_lag)
        
        # Calculate ACF and use moving block bootstrap for error estimation
        n_bootstrap = 100
        block_size = min(50, div(N, 10))
        
        for k in 1:max_lag
            lag = k - 1
            n_pairs = N - lag
            
            # Calculate C(lag) = E[(O[i] - mean) * (O[i+lag] - mean)]
            acf[k] = abs(mean(O_centered[i] * O_centered[i + lag] for i in 1:n_pairs))
            
            # Moving block bootstrap
            bootstrap_vals = zeros(n_bootstrap)
            for b in 1:n_bootstrap
                # Create bootstrap sample by selecting random starting points for blocks
                n_blocks = div(N, block_size)
                boot_sample = Float64[]
                
                for _ in 1:n_blocks
                    # Random starting point (circular: wrap around if needed)
                    start_idx = rand(1:N)
                    for j in 0:(block_size-1)
                        idx = mod1(start_idx + j, N)
                        push!(boot_sample, O[idx])
                    end
                end
                
                # Calculate ACF on bootstrap sample
                if length(boot_sample) > lag
                    boot_mean = mean(boot_sample)
                    boot_centered = boot_sample .- boot_mean
                    n_boot_pairs = min(length(boot_sample) - lag, n_pairs)
                    if n_boot_pairs > 0
                        boot_acf = abs(mean(boot_centered[i] * boot_centered[i + lag] for i in 1:n_boot_pairs))
                        bootstrap_vals[b] = boot_acf
                    end
                end
            end
            
            # Use standard deviation of bootstrap samples as error estimate
            acf_err[k] = std(bootstrap_vals)
        end
        
        @info "ACF calculated with bootstrap error estimation (n_bootstrap=$n_bootstrap, block_size=$block_size)"
        @info "acf[1] = $(acf[1]) ± $(acf_err[1]) (should be ~1.0)"
        @info "First few ACF values: $(acf[1:min(10, length(acf))])"
        
        # Exponential decay model: A*exp(-d/ξ)
        model(x, p) = p[1] .* exp.(-x ./ p[2])  # p[1]=A, p[2]=ξ
        lags_for_fit = collect(0:(max_lag-1))
        @show lags_for_fit
        acf_for_fit = acf
        
        # Initial guess for parameters
        # A should be around acf[1] (correlation at lag=0)
        A_init = acf[1]
        # ξ (correlation length) - estimate from where correlation drops to 1/e
        xi_init = 10.0  # Default guess
        for i in 2:length(acf)
            if acf[i] < acf[1] / ℯ
                xi_init = Float64(i - 1)
                break
            end
        end
        xi_init = max(1.0, min(xi_init, max_lag / 2.0))  # Bound initial guess
        p0 = [A_init, xi_init]
        
        # Fit the model
        fit_result = nothing
        A_fit = NaN
        xi_fit = NaN
        try
            fit_result = curve_fit(model, lags_for_fit, acf_for_fit, p0)
            fitted_params = coef(fit_result)
            A_fit = fitted_params[1]
            xi_fit = fitted_params[2]
            
            @info "Exponential decay fit results:"
            @info "  A (amplitude) = $(round(A_fit, digits=6))"
            @info "  ξ (correlation length) = $(round(xi_fit, digits=4))"
            
            # Compare with gap prediction
            gap_file = joinpath(dirname(filename), "$(measure_first)_first_gap_list_g=$(g).dat")
            if isfile(gap_file)
                gap_lines = readlines(gap_file)
                if !isempty(gap_lines)
                    gap = parse(Float64, gap_lines[end])
                    xi_predicted = 1.0 / gap
                    @info "  Predicted ξ from gap: $(round(xi_predicted, digits=4))"
                    @info "  Ratio ξ_fit/ξ_predicted: $(round(xi_fit/xi_predicted, digits=4))"
                end
            end
        catch e
            @warn "Exponential fit failed: $e"
        end
        
        # Create the plot with error bars and log scale
        lags = 0:(max_lag-1)
        p = Plots.plot(
            lags,
            acf,
            xlabel="d",
            ylabel="|⟨CᵢCᵢ₊d⟩|",
            title="$(measure_first) correlation for row=$row, g=$g",
            legend=:topright,
            seriestype=:scatter,
            size=(800, 600),
            marker=:circle,
            markersize=4,
            yerror=acf_err,
            label="Data with error bars",
            yaxis=:log,
            ylims=(1e-5, 1)
        )
        
 
        # Get gap for theoretical curve
        gap_file = joinpath(dirname(filename), "$(measure_first)_first_gap_list_g=$(g).dat")
        gap = NaN
        if isfile(gap_file)
            gap_lines = readlines(gap_file)
            if !isempty(gap_lines)
                gap = parse(Float64, gap_lines[end])
                xi_theory = 1.0 / gap
                @show gap, xi_theory
                # Add theoretical curve: A*exp(-d/ξ) with ξ from gap
                A_theory = acf[1]  # Use observed amplitude
                theoretical_curve = A_theory .* exp.(-lags ./ xi_theory)
                Plots.plot!(p, lags, theoretical_curve,
                    linewidth=2,
                    linestyle=:solid,
                    color=:green,
                    label="Theory: A exp(-d/$(round(xi_theory, digits=2))) (ξ from gap)"
                )
            end
        end
        
        # Add fitted curve if fit succeeded
        if fit_result !== nothing
            fitted_params = coef(fit_result)
            A_fit_plot = fitted_params[1]
            xi_fit_plot = fitted_params[2]
            fitted_curve = model(lags, fitted_params)
            Plots.plot!(p, lags, fitted_curve, 
                linewidth=2, 
                linestyle=:dash, 
                color=:red,
                label="Fit: $(round(A_fit_plot, digits=3)) exp(-d/$(round(xi_fit_plot, digits=2)))"
        )
        end
    
        # Add horizontal line at 0
        Plots.hline!(p, [0], linestyle=:dot, color=:gray, linewidth=1, label="")
        
        # Save the plot
        if save_path === nothing
            save_path = "image/$(measure_first)_ACF_row=$(row)_g=$(g).pdf"
        end
        savefig(p, save_path)
        @info "Plot saved as '$save_path'"
        
        return (acf, p, fit_result)
        
    catch e
        @error "Error processing file $filename: $e"
        return nothing
    end
end

function spin_correlation(J::Float64, g::Float64, p::Int,row::Int, nqubits::Int; measure_first=:Z, conv_step=1000, samples=100000, data_dir="data", save_path=nothing, max_lag=nothing, hor=true)
    filename = joinpath(data_dir, "compile_final_params_row=$(row)_g=$(g).dat")
    
    if !isfile(filename)
        @error "File not found: $filename"
        return nothing
    end
    @info "Reading $filename..."
    
    # Read all data from file and form a vector
    params = []
    data_lines = []
    
    open(filename, "r") do file
        for line in eachline(file)
            stripped = strip(line)
            if !startswith(stripped, "#") && !isempty(stripped)
                push!(data_lines, line)
                param_elements = parse.(Float64, split(stripped))
                push!(params, param_elements)
            end
        end
    end
        
    if isempty(params)
        @error "No data found in $filename"
            return nothing
    end
    params = vcat(params...)
    A_matrix = build_gate_from_params(params, p, row, nqubits; share_params=true)
    rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=conv_step, samples=samples,measure_first=measure_first)
    Z_list = Z_list[conv_step:end]
    energy = energy_measure(X_list, Z_list, g, J, row)
    @show energy

    # Build correlation matrix
    corr_matrix = zeros(row, row)
    
    # Diagonal elements (self-correlation)
    for i in 1:row
        corr_matrix[i, i] = mean(Z_list[i:row:end] .^ 2)
    end
    
    # Off-diagonal elements (cross-correlation)
    # Calculate endpoint that gives equal length arrays for all starting indices
    N = length(Z_list)
    max_steps = div(N - row, row)  # Maximum number of complete steps
    endpoint = row + max_steps * row  # Common endpoint for all indices 1:row
    @show endpoint
    for i in 1:row
        for j in (i+1):row
            corr_value = mean(Z_list[i:row:endpoint] .* Z_list[j:row:endpoint])
            corr_matrix[i, j] = corr_value
            corr_matrix[j, i] = corr_value  # Symmetric matrix
            println("Z$(i)$(j)_mean = $corr_value")
        end
    end
    
    # Create heatmap
    p = Plots.heatmap(1:row, 1:row, corr_matrix,
        xlabel="σᵢᶻ", ylabel="σⱼᶻ",
        color=cgrad(:coolwarm, rev=true),
        clims=(0.0, 1.0),
        aspect_ratio=:equal,
        size=(500, 450),
        colorbar=true,
        title="Spin Correlation Matrix (g=$g)",
        xticks=1:row, yticks=1:row)
    
    # Save plot if save_path is provided
    if save_path == nothing
        save_path = "image/spin_correlation_row=$(row)_g=$(g).pdf"
        savefig(p, save_path)
        @info "Correlation matrix saved to $save_path"
    end
    
    display(p)
    return corr_matrix
end

function C1C2(g_values::Vector{Float64}, J::Float64, p::Int, row::Int, nqubits::Int; measure_first=:Z, conv_step=1000, samples=1000, data_dir="data")
    
    # Store results
    results = Dict("g" => Float64[], "C1" => Float64[], "C2" => Float64[], "energy" => Float64[])
    
    for g in g_values   
        filename = joinpath(data_dir, "compile_final_params_p=$(p)_row=$(row)_nqubits=$(nqubits)_g=$(g).dat")
        if !isfile(filename)
            @error "File not found: $filename"
            continue
        end
        @info "Processing g=$g..."
        
        # Read all data from file and form a vector
        params = []
        data_lines = []
        
        open(filename, "r") do file
            for line in eachline(file)
                stripped = strip(line)
                if !startswith(stripped, "#") && !isempty(stripped)
                    push!(data_lines, line)
                    param_elements = parse.(Float64, split(stripped))
                    push!(params, param_elements)
                end
            end
        end
            
        if isempty(params)
            @error "No data found in $filename"
            continue
        end
        params = vcat(params...)

        A_matrix = build_gate_from_params(params, p, row, nqubits; share_params=true)
        rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=conv_step, samples=samples, measure_first=measure_first)
        Z_list = Z_list[conv_step:end]
        energy = energy_measure(X_list, Z_list, g, J, row)
        
        # Calculate connected correlations
        # For this, we need to compute <Z_i>, <Z_j>, <Z_i * Z_j>
        N = length(Z_list)
        max_steps = div(N - row, row)
        endpoint = row + max_steps * row
        
        # Compute single-site expectation values
        Z1_mean = mean(Z_list[1:row:endpoint])
        Z2_mean = mean(Z_list[2:row:endpoint])
        Z3_mean = row >= 3 ? mean(Z_list[3:row:endpoint]) : 0.0
        
        # Compute two-point correlations
        Z1Z2_mean = mean(Z_list[1:row:endpoint] .* Z_list[2:row:endpoint])
        Z1Z3_mean = row >= 3 ? mean(Z_list[1:row:endpoint] .* Z_list[3:row:endpoint]) : 0.0
        
        # Compute connected correlations
        C1 = abs(Z1Z2_mean - Z1_mean * Z2_mean) # Nearest neighbor
        C2 = row >= 3 ? abs(Z1Z3_mean - Z1_mean * Z3_mean) : 0.0  # Next-nearest neighbor
        
        # Store results
        push!(results["g"], g)
        push!(results["C1"], C1)
        push!(results["C2"], C2)
        push!(results["energy"], energy)
        
        @info "g=$g: Energy=$energy, C1=$C1, C2=$C2"
    end
    
    # Create plots using CairoMakie
    if !isempty(results["g"])
        # Sort by g values for better visualization
        sort_idx = sortperm(results["g"])
        g_sorted = results["g"][sort_idx]
        C1_sorted = results["C1"][sort_idx]
        C2_sorted = results["C2"][sort_idx]
        
        # Plot C1 vs g
        fig1 = Figure(size=(800, 600))
        ax1 = Axis(fig1[1, 1], 
                   xlabel="g", 
                   ylabel="C1 (Nearest Neighbor Connected Correlation)",
                   yscale=log10,
                   title="Connected Correlation C1 vs g (row=$row)")
        CairoMakie.ylims!(ax1, 10^(-5), 1)
        lines!(ax1, g_sorted, C1_sorted, linewidth=2, color=:blue)
        CairoMakie.scatter!(ax1, g_sorted, C1_sorted, markersize=12, color=:blue)
        
        # Save C1 plot
        save_path1 = "image/C1_vs_g_row=$(row).pdf"
        save(save_path1, fig1)
        @info "C1 plot saved to $save_path1"
        
        # Plot C2 vs g
        fig2 = Figure(size=(800, 600))
        ax2 = Axis(fig2[1, 1], 
                   xlabel="g", 
                   ylabel="C2 (Next-Nearest Neighbor Connected Correlation)",
                   yscale=log10,
                   title="Connected Correlation C2 vs g (row=$row)")
                   CairoMakie.ylims!(ax2, 10^(-5), 1)
        lines!(ax2, g_sorted, C2_sorted, linewidth=2, color=:red)
        CairoMakie.scatter!(ax2, g_sorted, C2_sorted, markersize=12, color=:red)
        
        # Save C2 plot
        save_path2 = "image/C2_vs_g_row=$(row).pdf"
        save(save_path2, fig2)
        @info "C2 plot saved to $save_path2"
        
        # Combined plot
        fig3 = Figure(size=(800, 600))
        ax3 = Axis(fig3[1, 1], 
                   xlabel="g", 
                   ylabel="Connected Correlation",
                   yscale=log10,
                   title="Connected Correlations vs g (row=$row)")
                   CairoMakie.ylims!(ax3, 10^(-5), 1)
        lines!(ax3, g_sorted, C1_sorted, linewidth=2, color=:blue, label="C1 (nearest)")
        CairoMakie.scatter!(ax3, g_sorted, C1_sorted, markersize=12, color=:blue)
        lines!(ax3, g_sorted, C2_sorted, linewidth=2, color=:red, label="C2 (next-nearest)")
        CairoMakie.scatter!(ax3, g_sorted, C2_sorted, markersize=12, color=:red)
        axislegend(ax3, position=:rt)
        
        # Save combined plot
        save_path3 = "image/C1C2_vs_g_row=$(row).pdf"
        save(save_path3, fig3)
        @info "Combined C1C2 plot saved to $save_path3"
        
        display(fig1)
        display(fig2)
        display(fig3)
    else
        @warn "No data to plot"
    end
    
    return results
end

function correlation(g::Float64,row::Int; measure_first=:X, data_dir="data", save_path=nothing, max_lag=nothing)
    
    filename = joinpath(data_dir, "compile_Z_list_list_row=$(row)_g=$(g).dat")
    
    if !isfile(filename)
        @error "File not found: $filename"
        return nothing
    end
    @info "Reading $filename..."
    
    # Read the last line
    try
        data_lines = []
        open(filename, "r") do file
            for line in eachline(file)
                if !startswith(strip(line), "#") && !isempty(strip(line))
                    push!(data_lines, line)
                end
            end
        end
        
        if isempty(data_lines)
            @error "No data lines found in $filename"
            return nothing
        end
        
        # Get the last line
        last_line = data_lines[end]
        
        O = [parse(Float64, x) for (i, x) in enumerate(split(last_line)) if i % 3 != 0]
        #O = [parse(Float64, x) for (i, x) in enumerate(split(last_line)) if i % 4 == 0]
        #O = parse.(Float64, split(last_line))
        O1 = O[1:2:end]
        O2 = O[2:2:end]
       N = length(O)
        @info "g=$g: Found $(length(O1)) and $(length(O2)) observable values in last line"
        
        if N < 10
            @error "g=$g: Too few data points ($N) for autocorrelation analysis"
            return nothing
        end
        
        # Determine maximum lag
        if max_lag === nothing
            max_lag = min(500, div(N, 2))
        else
            max_lag = min(max_lag, div(N, 2))
        end
        
        @info "Calculating autocorrelation function up to lag $max_lag"
        
        # Calculate mean and variance (using every other point: 1, 3, 5, ...)
        O_mean = mean(O)
        O_centered = O .- O_mean

        
        # Calculate autocorrelation function manually
        acf = zeros(max_lag)
        
        for k in 1:max_lag
            lag = k - 1
            acf[1] = (abs(mean(O1 .^2) - mean(O1)^2) + abs(mean(O2 .^2) - mean(O2)^2))/2
            acf[2] = abs(mean(O1 .* O2) - mean(O1) * mean(O2))
        end
    
        @info "First few ACF values: $(acf[1:min(10, length(acf))])"
        
        # Fit exponential model: exp(-x / ξ)
        model(x, p) = p[1] .* exp.(-x ./ p[2])
        
        # Use lags >= 0 for fitting
        lags_for_fit = collect(0:(max_lag-1))
        acf_for_fit = acf
        
        # Initial guess: A ≈ acf[1], ξ ≈ -1/log(|acf[2]/acf[1]|)
        A_init = acf[1]
        ratio_raw = length(acf) >= 2 && abs(A_init) > eps() ? abs(acf[2] / A_init) : 0.5
        ratio = clamp(ratio_raw, 1e-6, 0.999)
        xi_init = -1.0 / log(ratio)
        xi_init = max(0.1, min(xi_init, 100.0))  # Bound initial guess
        p0 = [A_init, xi_init]
        
        # Fit the model
        fit_result = nothing
        try
            fit_result = curve_fit(model, lags_for_fit, acf_for_fit, p0)
            fitted_params = coef(fit_result)
            A_fit, xi_fit = fitted_params
            
            @info "Exponential fit results:"
            @info "  ξ (correlation length) = $(round(xi_fit, digits=4))"
            
            # Compare with gap prediction
            gap_file = joinpath(dirname(filename), "$(measure_first)_first_gap_list_g=$(g).dat")
            if isfile(gap_file)
                gap_lines = readlines(gap_file)
                if !isempty(gap_lines)
                    gap = parse(Float64, gap_lines[end])
                    xi_predicted = 1.0 / gap
                    @info "  Predicted ξ from gap: $(round(xi_predicted, digits=4))"
                    @info "  Ratio ξ_fit/ξ_predicted: $(round(xi_fit/xi_predicted, digits=4))"
                end
            end
        catch e
            @warn "Exponential fit failed: $e"
        end
        
        # Create the plot
        lags = 0:(max_lag-1)
        xi_str = isnan(xi_fit) ? "n/a" : string(round(xi_fit, digits=2))
        p = Plots.plot(
            lags,
            acf,
            xlabel="Lag k",
            ylabel="Autocorrelation C(k)",
            title="$(measure_first) Autocorrelation for g=$g, ξ_fit=$(xi_str)",
            legend=:topright,
            linewidth=2,
            size=(800, 600),
            marker=:circle,
            markersize=3,
            label="ACF (data)"
        )
        
        # Add fitted curve if fit succeeded
        if fit_result !== nothing
            fitted_params = coef(fit_result)
            A_fit, xi_fit = fitted_params
            fitted_curve = model(lags, fitted_params)
            Plots.plot!(p, lags, fitted_curve, 
                linewidth=2, 
                linestyle=:dash, 
                color=:red,
                label="Fit: $(round(A_fit, digits=2))·exp(-k/$(round(xi_fit, digits=2)))"
        )
        end
        
        # Add horizontal line at 0
        Plots.hline!(p, [0], linestyle=:dot, color=:gray, linewidth=1, label="")
        
        # Save the plot
        if save_path === nothing
            save_path = "image/$(measure_first)_ACF_g=$(g).pdf"
        end
        savefig(p, save_path)
        @info "Plot saved as '$save_path'"
        
        return (acf, p, fit_result)
        
    catch e
        @error "Error processing file $filename: $e"
        return nothing
    end
end

function magnectization(g_values::Vector{Float64}, row::Int; data_dir="data")
    # Store results
    results = Dict("g" => Float64[], "Z_mean" => Float64[], "X_mean" => Float64[])
    
    for g in g_values
        X_filename = joinpath(data_dir, "compile_X_list_list_row=$(row)_g=$(g).dat")
        Z_filename = joinpath(data_dir, "compile_Z_list_list_row=$(row)_g=$(g).dat")
        
        # Check if both files exist
        if !isfile(X_filename)
            @error "File not found: $X_filename"
            continue
        end
        if !isfile(Z_filename)
            @error "File not found: $Z_filename"
            continue
        end
        
        @info "Processing g=$g..."
        
        # Read X_list from last line (converged values)
        X_list = nothing
        try
            open(X_filename, "r") do file
                lines = readlines(file)
                data_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), lines)
                if isempty(data_lines)
                    @error "No data found in $X_filename"
            return nothing
        end
                X_list = parse.(Float64, split(data_lines[end]))
            end
        catch e
            @error "Error reading $X_filename: $e"
            continue
        end
        
        # Read Z_list from last line (converged values)
        Z_list = nothing
        try
            open(Z_filename, "r") do file
                lines = readlines(file)
                data_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), lines)
                if isempty(data_lines)
                    @error "No data found in $Z_filename"
                    return nothing
                end
                Z_list = parse.(Float64, split(data_lines[end]))
            end
        catch e
            @error "Error reading $Z_filename: $e"
            continue
        end
        
        # Calculate mean values
        X_mean = abs(mean(X_list))
        Z_mean = abs(mean(Z_list))
        
        # Store results
        push!(results["g"], g)
        push!(results["X_mean"], X_mean)
        push!(results["Z_mean"], Z_mean)
        
        @info "g=$g: <X>=$X_mean, <Z>=$Z_mean"
    end
    
    if isempty(results["g"])
        @error "No valid data found"
        return nothing
    end
    
    # Create X magnetization plot
    fig_X = Figure(size=(800, 600))
    ax_X = Axis(fig_X[1, 1], 
                xlabel="g", 
                ylabel="<X> Magnetization",
                title="Mean <X> Magnetization vs g (row=$row)")
    
    lines!(ax_X, results["g"], results["X_mean"], linewidth=2, color=:blue)
    CairoMakie.scatter!(ax_X, results["g"], results["X_mean"], markersize=12, color=:blue)
    
    # Save X plot
    save_path_X = "image/X_magnetization_vs_g_row=$(row).pdf"
    save(save_path_X, fig_X)
    @info "X magnetization plot saved to $save_path_X"
    display(fig_X)
    
    # Create Z magnetization plot
    fig_Z = Figure(size=(800, 600))
    ax_Z = Axis(fig_Z[1, 1], 
                xlabel="g", 
                ylabel="<Z> Magnetization",
                title="Mean <Z> Magnetization vs g (row=$row)")
    
    lines!(ax_Z, results["g"], results["Z_mean"], linewidth=2, color=:red)
    CairoMakie.scatter!(ax_Z, results["g"], results["Z_mean"], markersize=12, color=:red)
    
    # Save Z plot
    save_path_Z = "image/Z_magnetization_vs_g_row=$(row).pdf"
    save(save_path_Z, fig_Z)
    @info "Z magnetization plot saved to $save_path_Z"
    display(fig_Z)
    
    return results
end

function dynamics(g::Float64, J::Float64, p::Int, row::Int, nqubits::Int; conv_step=1000, samples=1000, nshots=100, data_dir="data")
    filename = joinpath(data_dir, "compile_final_params_row=$(row)_g=$(g).dat")
    
    if !isfile(filename)
        @error "File not found: $filename"
        return nothing
    end
    @info "Reading $filename..."
    
    # Read parameters from file
    params = []
    try
        open(filename, "r") do file
            for line in eachline(file)
                stripped = strip(line)
                if !startswith(stripped, "#") && !isempty(stripped)
                    param_elements = parse.(Float64, split(stripped))
                    push!(params, param_elements)
                end
            end
        end
        
        if isempty(params)
            @error "No data found in $filename"
            return nothing
        end
        params = vcat(params...)
    catch e
        @error "Error reading $filename: $e"
        return nothing
    end
    
    # ===== X Dynamics (measure_first = :X) =====
    @info "Running X dynamics simulation with $nshots shots and $samples samples (measure_first=:X)..."
    
    X_dynamics = []  # Each element is a vector of X values over time for one shot
    
    for shot in 1:nshots
        if shot % 10 == 0
            @info "Processing X dynamics shot $shot/$nshots..."
        end
        
        # Build gate and run simulation with measure_first=:X
        A_matrix = build_gate_from_params(params, p, row, nqubits; share_params=true)
        rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=conv_step, samples=samples, measure_first=:X)
        
        push!(X_dynamics, X_list)
    end
    
    @info "X dynamics simulation completed. Processing results..."
    
    # Calculate mean and std over all shots at each time step for X
    n_steps_X = minimum(length.(X_dynamics))
    steps_X = 1:n_steps_X
    
    X_mean = [mean([X_dynamics[shot][step] for shot in 1:nshots]) for step in 1:n_steps_X]
    X_std = [std([X_dynamics[shot][step] for shot in 1:nshots]) for step in 1:n_steps_X]
    
    # Plot X dynamics
    fig_X = Figure(size=(800, 600))
    ax_X = Axis(fig_X[1, 1], 
                xlabel="Time Step", 
                ylabel="⟨X⟩",
                title="X Magnetization Dynamics (g=$g, row=$row)")
    
    lines!(ax_X, steps_X, X_mean, linewidth=2, color=:blue)
    #band!(ax_X, steps_X, X_mean .- X_std, X_mean .+ X_std, alpha=0.3, color=:blue)
    
    save_path_X = "image/dynamics_X_g=$(g)_row=$(row).pdf"
    save(save_path_X, fig_X)
    @info "X dynamics plot saved to $save_path_X"
    display(fig_X)
    
    # ===== Z Dynamics (measure_first = :Z) =====
    @info "Running Z dynamics simulation with $nshots shots and $samples samples (measure_first=:Z)..."
    
    Z_dynamics = []  # Each element is a vector of Z values over time for one shot
    
    for shot in 1:nshots
        if shot % 10 == 0
            @info "Processing Z dynamics shot $shot/$nshots..."
        end
        
        # Build gate and run simulation with measure_first=:Z
        A_matrix = build_gate_from_params(params, p, row, nqubits; share_params=true)
        rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=conv_step, samples=samples, measure_first=:Z)
      
        push!(Z_dynamics, Z_list)
    end
    
    @info "Z dynamics simulation completed. Processing results..."
    
    # Calculate mean and std over all shots at each time step for Z
    n_steps_Z = minimum(length.(Z_dynamics))
    steps_Z = 1:n_steps_Z
    
    Z_mean = [mean([Z_dynamics[shot][step] for shot in 1:nshots]) for step in 1:n_steps_Z]
    Z_std = [std([Z_dynamics[shot][step] for shot in 1:nshots]) for step in 1:n_steps_Z]
    
    # Plot Z dynamics
    fig_Z = Figure(size=(800, 600))
    ax_Z = Axis(fig_Z[1, 1], 
                xlabel="Time Step", 
                ylabel="⟨Z⟩",
                title="Z Magnetization Dynamics (g=$g, row=$row)")
    
    lines!(ax_Z, steps_Z, Z_mean, linewidth=2, color=:red)
    #band!(ax_Z, steps_Z, Z_mean .- Z_std, Z_mean .+ Z_std, alpha=0.3, color=:red)
    
    save_path_Z = "image/dynamics_Z_g=$(g)_row=$(row).pdf"
    save(save_path_Z, fig_Z)
    @info "Z dynamics plot saved to $save_path_Z"
    display(fig_Z)
    
    # Return results
    results = Dict(
        "X_steps" => collect(steps_X),
        "X_mean" => X_mean,
        "X_std" => X_std,
        "Z_steps" => collect(steps_Z),
        "Z_mean" => Z_mean,
        "Z_std" => Z_std
    )
    
    return results
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
            save_path = "image/X_block_var_g=$(g_str)_lines_$(lines_str).pdf"
        end
        savefig(p, save_path)
        @info "Figure saved to: $save_path"
    end
    
    return p
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

function draw()
    g_list = [1.0, 1.5,2.0,2.5,3.0,3.5,4.0]
    MPSKit_list = [1/0.2578353678314149, 1/ 0.33555540204090634, 1/0.4449543312401218, 1/0.6647535608179987, 1/1.4604857883943378, 1/1.020585926250979, 1/0.8310444496509529]
    contractrow2_list = [4.229341868437619, 3.2027058878118466, 2.5141837897218506, 1.5644736638192145, 1.0340527262643096, 1.2982558625936602, 1.5644736638192145]
    contractrow3_list = [6.72214645054244, 3.223771967755304, 2.0509835248150377, 3.203333854222672, 0.031102147204431205, 1.7399907753796913, 2.127230859908191]
    measurerow2_list = [3.7293788678313327, 3.056366958451286, 2.6882821633610874, 3.1883266192455997, 1.4256636831433547, 1.9737348169983167, 2.6516581706512636]
    measurerow3_list = [4.1658553940994207, 3.9027639250642845, 4.018628504686564, 3.9956480223211437, 2.617924888881849, 2.6665735844616196, 3.9787566486882997]
    corr_length= [1/0.26,1/0.26,1/0.26,1/0.36,1/0.4]

    # Create figure and axis with CairoMakie
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
              xlabel="1/row",
              ylabel="1/λ",
              title="1/λ vs 1/row, g=2.0",
              limits=(0.0, 0.5, 0.0, 5.0),
              xticks=[0.0, 0.25, 0.5],
              yticks=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
              xgridvisible=true,
              ygridvisible=true,
              xgridcolor=(:gray, 0.3),
              ygridcolor=(:gray, 0.3))

    # Plot data
    x_data = [1/2, 1/4, 1/6, 1/8, 1/10]
    scatterlines!(ax, x_data, corr_length,
                  label="1/λ",
                  color=(:orange, 0.5),
                  marker=:star5,
                  markersize=15,
                  linewidth=2)
    
    #=
    scatterlines!(ax, g_list, contractrow2_list,
                  label="contractrow2",
                  color=(:green, 0.7),
                  marker=:xcross,
                  markersize=10,
                  linewidth=2)
    scatterlines!(ax, g_list, contractrow3_list,
                  label="contractrow3",
                  color=(:orange, 0.5),
                  marker=:star5,
                  markersize=12,
                  linewidth=2)
    scatterlines!(ax, g_list, measurerow2_list,
                  label="measurerow2",
                  color=(:blue, 0.5),
                  marker=:diamond,
                  markersize=10,
                  linewidth=2)
    scatterlines!(ax, g_list, measurerow3_list,
                  label="measurerow3",
                  color=(:black, 0.5),
                  marker=:circle,
                  markersize=10,
                  linewidth=2)
    =#
    
    # Add legend
    axislegend(ax, position=:rt, labelsize=10)
    
    # Save and display
    save("image/scaling.pdf", fig)
    display(fig)
    
    return fig
end

function draw_energy_error()
    g_list = [1.0, 1.5,2.0,2.5,3.0,3.5,4.0]
    MPSKit = [-2.125426272074635, -2.283531518020085, -2.507866896802187, -2.8478757938149273, -3.2682943897391175, -3.7190408503688848, -4.185357190258307]
    row2 = [-2.1229893589358937, -2.3477761576157614, -2.5767595159515952, -2.898724672467247, -3.3154599259925988, -3.7435440244024405, -4.221634083408341]
    row2_error = abs.(MPSKit .- row2)
    row3 = [-2.121685550044853, -2.341473962110339, -2.576367881285688, -2.8996361592027333, -3.254721804738692, -3.6923713696636424, -4.1231005899660405]
    row3_error = abs.(MPSKit .- row3)

    fig = Plots.plot(xlabel="g", ylabel="energy error", 
                     title="energy error vs Transverse Field (2D)", 
                     xticks=[1.0, 1.5,2.0,2.5,3.0,3.5,4.0], size=(800, 600))

    colors = [RGBA(1,0,0,0.5), RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), 
              RGBA(0,0,1,0.5), RGBA(0,0,0,0.5)]
    markers = [:+, :x, :star, :diamond, :circle]
    
    #Plots.plot!(fig, g_list, MPSKit_list, label="MPSKit",
              #color=colors[1], marker=markers[1], markersize=1, linewidth=2)
    Plots.plot!(fig, g_list, row2_error, label="row2",
                color=colors[2], marker=markers[2], markersize=2, linewidth=2)
    Plots.plot!(fig, g_list, row3_error, label="row3",
                color=colors[3], marker=markers[3], markersize=3, linewidth=2)

    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:topright, legendfontsize=10)
    Plots.savefig(fig, "image/energy_error_vs_g_2D.pdf")
    Plots.display(fig)
                
    return fig
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

function energy_con(g::Union{Float64, Vector{Float64}}, row::Int; data_dir="data")
    # Convert single value to vector for uniform processing
    g_values = g isa Float64 ? [g] : g
    
    if isempty(g_values)
        @error "No g values provided"
        return nothing
    end
    
    # Exact energy values for different g values
    # g values: 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0
    g_exact = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
    energy_exact = [-1.9999999999999971, -2.0078141005791723, -2.031275809659576, -2.0704448852547266, 
                    -2.125426272074635, -2.1963790626176034, -2.283531518020085, -2.3872064546848364, 
                    -2.507866896802187, -2.669431612922372, -2.8478757938149273, -3.052726110722211, 
                    -3.2682943897391175]
    
    # Create dictionary for easy lookup
    exact_energy_dict = Dict(zip(g_exact, energy_exact))
    
    # Initialize plot
    p = nothing
    colors = palette(:tab10)
    valid_plots = 0
    max_iterations = 0
    
    for (idx, g_val) in enumerate(g_values)
        # Construct filename
        filename = joinpath(data_dir, "compile_energy_history_row=$(row)_g=$(g_val)_.dat")
        
        if !isfile(filename)
            @warn "File not found: $filename (skipping)"
            continue
        end
        @info "Reading $filename..."
        
        # Read energy data, skipping comment lines
        energy_data = Float64[]
        open(filename, "r") do file
            for line in eachline(file)
                stripped = strip(line)
                # Skip comments and empty lines
                if !startswith(stripped, "#") && !isempty(stripped)
                    try
                        energy = parse(Float64, stripped)
                        push!(energy_data, energy)
                    catch e
                        @warn "Could not parse line: $line"
                    end
                end
            end
        end
        
        if isempty(energy_data)
            @warn "No valid energy data found in $filename (skipping)"
            continue
        end

        n_iterations = length(energy_data)
        max_iterations = max(max_iterations, n_iterations)
        @info "Found $n_iterations energy values for g=$(g_val)"
        
        # Create or add to plot
        if p === nothing
            # First valid plot - initialize
            p = Plots.plot(
                1:n_iterations,
                energy_data,
                xlabel="Iteration",
                ylabel="Energy",
                title="Energy Convergence, row=$row",
                label="g=$(g_val)",
                linewidth=2,
                size=(800, 600),
                color=colors[mod1(idx, length(colors))],
                legend=length(g_values) > 1 ? :topright : false
            )
        else
            # Add to existing plot
            Plots.plot!(p,
                1:n_iterations,
                energy_data,
                label="g=$(g_val)",
                linewidth=2,
                color=colors[mod1(idx, length(colors))]
            )
        end
        
        # Add exact energy line if available
        if haskey(exact_energy_dict, g_val)
            E_exact = exact_energy_dict[g_val]
            Plots.hline!(p,
                [E_exact],
                label="g=$(g_val) exact",
                linewidth=1.5,
                linestyle=:dash,
                color=colors[mod1(idx, length(colors))],
                alpha=0.7
            )
        end
        
        valid_plots += 1
    end
    
    if p === nothing
        @error "No valid data found for any g value"
        return nothing
    end
    
    # Save the plot
    if length(g_values) == 1
        save_path = "image/energy_convergence_g=$(g_values[1])_row=$row.pdf"
    else
        g_str = join(g_values, "_")
        save_path = "image/energy_convergence_g=$(g_str)_row=$row.pdf"
    end
    savefig(p, save_path)
    @info "Figure saved to: $save_path"
    
    return p
end

function var_samples(g::Float64, J::Float64, row::Int; data_dir="data", save_path=nothing, sample_sizes=100:100:10000)
    # Read X data
    X_filename = joinpath(data_dir, "Z_first_$(row)_X_list_list_g=$(g).dat")
    if !isfile(X_filename)
        @error "File not found: $X_filename"
        return nothing
    end
    
    # Read Z data
    Z_filename = joinpath(data_dir, "Z_first_$(row)_Z_list_list_g=$(g).dat")
    if !isfile(Z_filename)
        @error "File not found: $Z_filename"
        return nothing
    end
    
    @info "Reading data files..."
    
    # Read last line of X data
    X_list = nothing
    open(X_filename, "r") do file
        lines = readlines(file)
        data_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), lines)
        if isempty(data_lines)
            @error "No data found in $X_filename"
            return nothing
        end
        X_list = parse.(Float64, split(data_lines[end]))
    end
    
    # Read last line of Z data
    Z_list = nothing
    open(Z_filename, "r") do file
        lines = readlines(file)
        data_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), lines)
        if isempty(data_lines)
            @error "No data found in $Z_filename"
            return nothing
        end
        Z_list = parse.(Float64, split(data_lines[end]))
    end
    
    # Use the minimum length to ensure both lists have data
    n_samples = min(length(X_list), length(Z_list))
    @info "Found $(length(X_list)) X samples and $(length(Z_list)) Z samples"
    @info "Using minimum length: $n_samples samples"
    
    # Truncate to minimum length
    X_list = X_list[1:n_samples]
    Z_list = Z_list[1:n_samples]
    
    # Calculate Hamiltonian for each sample
    # H = -g ∑ᵢ Xᵢ - J ∑ᵢ ZᵢZᵢ₊₁ - J ∑ᵢ ZᵢZᵢ₊ᵣₒᵥ
    # Note: X starts from index 1, Z calculations start from index 1001
    H_list = Float64[]
    
    @info "Calculating Hamiltonian for each sample (X from 1, Z from 1001)..."
    Z_start_idx = 1001
    
    if n_samples < Z_start_idx + row
        @error "Not enough samples (need at least $(Z_start_idx + row), have $n_samples)"
        return nothing
    end
    
    for i in 1:n_samples
        # X term: -g * X[i] (X starts from index 1)
        H_X = -g * X_list[i]
        
        # ZZ nearest neighbor term: -J * Z[i] * Z[i+1]
        # Only calculate if i >= Z_start_idx and i+1 is within bounds
        H_ZZ_neighbor = 0.0
        if i >= Z_start_idx && i + 1 <= n_samples
            H_ZZ_neighbor = -J * Z_list[i] * Z_list[i + 1]
        end
        
        # ZZ row term: -J * Z[i] * Z[i+row]
        # Only calculate if i >= Z_start_idx and i+row is within bounds
        H_ZZ_row = 0.0
        if i >= Z_start_idx && i + row <= n_samples
            H_ZZ_row = -J * Z_list[i] * Z_list[i + row]
        end
        
        H = H_X + H_ZZ_neighbor + H_ZZ_row
        push!(H_list, H)
    end
    
    @info "Calculated $(length(H_list)) Hamiltonian values (X: 1 to $n_samples, Z: $Z_start_idx to $n_samples)"
    
    # Calculate variance for different sample sizes
    sample_size_list = Int[]
    variance_list = Float64[]
    std_error_list = Float64[]  # Standard error of variance estimate
    
    @info "Calculating variance for different sample sizes..."
    n_H_samples = length(H_list)
    @show n_H_samples
    for sample_size in sample_sizes
        if sample_size > n_H_samples
            @warn "Sample size $sample_size exceeds available data ($n_H_samples), stopping"
            break
        end
        
        # Calculate how many complete blocks we can form
        n_blocks = div(n_H_samples, sample_size)
        
        if n_blocks < 2
            @warn "Not enough data for sample size $sample_size (need at least 2 blocks), skipping"
            continue
        end
        
        # Calculate mean energy for each block
        block_means = Float64[]
        for block_idx in 1:n_blocks
            block_start = (block_idx - 1) * sample_size + 1
            block_end = block_idx * sample_size
            block_mean = mean(H_list[block_start:block_end])
            push!(block_means, block_mean)
        end
        
        # Calculate variance across blocks
        block_variance = var(block_means)
        
        # Calculate standard error (std of variance estimate)
        # For variance estimate: SE ≈ sqrt(2/(n-1)) * var
        std_error = sqrt(2.0 / (n_blocks - 1)) * block_variance
        
        push!(sample_size_list, sample_size)
        push!(variance_list, block_variance)
        push!(std_error_list, std_error)
        
        @info "Sample size: $sample_size, n_blocks: $n_blocks, variance: $block_variance"
    end
    
    if isempty(sample_size_list)
        @error "No valid sample sizes to plot"
        return nothing
    end
    
    # Create the plot
    p = Plots.plot(
        sample_size_list,
        variance_list,
        xlabel="Sample Size",
        ylabel="Variance",
        title="Variance vs Sample Size (g=$g, J=$J, row=$row)",
        legend=false,
        linewidth=2,
        marker=:circle,
        markersize=4,
        size=(800, 600),
        grid=true
    )
    
    # Add error bars
    Plots.plot!(p, sample_size_list, variance_list,
               ribbon=std_error_list,
               fillalpha=0.3,
               linewidth=2)
    
    # Add 1/sqrt(N) reference line
    N_range = [minimum(sample_size_list), maximum(sample_size_list)]
    reference_scale = variance_list[1] * sqrt(sample_size_list[1])
    reference_line = [reference_scale / sqrt(N) for N in N_range]
    Plots.plot!(p, N_range, reference_line,
               linestyle=:dash,
               linewidth=2,
               color=:red,
               label="1/√N scaling")
    
    # Save the figure
    if save_path === nothing
        save_path = "image/variance_vs_samples_g=$(g)_J=$(J)_row=$(row).pdf"
    end
    
    savefig(p, save_path)
    @info "Figure saved to: $save_path"
    
    return p, sample_size_list, variance_list, std_error_list
end

function var_mean_samples(g::Float64, J::Float64, row::Int, p::Int, nqubits::Int; data_dir="data", save_path=nothing, nshots=100)
    params_file = joinpath(data_dir, "compile_final_params_row=$(row)_g=$(g).dat")
    params = []
    open(params_file, "r") do file
        for line in eachline(file)
            if !startswith(strip(line), "#") && !isempty(strip(line))
                param_vec = parse.(Float64, split(line))
                push!(params, param_vec)
            end
        end
    end
    params = vcat(params...)
    @show params
    @info "Computing variance vs samples for g=$g, J=$J, row=$row with $(Threads.nthreads()) threads"
    _, exact_energy = exact_E_from_params(params, g, J, p, row, nqubits; data_dir="data", optimizer=GreedyMethod())
    
    # Define sample range
    sample_range = 1000:1000:10000
    n_samples = length(sample_range)
    
    # Pre-allocate results arrays
    var_list = Vector{Float64}(undef, n_samples)
    samples_list = collect(sample_range)
    
    # Parallelize over different sample sizes
    Threads.@threads for idx in 1:n_samples
        samples = samples_list[idx]
        @info "Thread $(Threads.threadid()): Processing samples=$samples"
        
        # For each sample size, compute nshots energies
        energy_list = Vector{Float64}(undef, nshots)
        for i in 1:nshots
            A_matrix = build_gate_from_params(params, p, row, nqubits)
            rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=1000, samples=samples, measure_first=:Z)
            energy = energy_measure(X_list, Z_list[1000:end], g, J, row) 
            energy_list[i] = energy
        end
        
        #variance = var(energy_list)
        variance = mean((energy_list .-(exact_energy)) .^2)
        var_list[idx] = variance
        @info "Thread $(Threads.threadid()): samples=$samples, variance=$variance"
    end
    
    # Create the plot
    p_plot = Plots.plot(
        samples_list,
        var_list,
        xlabel="Number of Samples",
        ylabel="Variance",
        title="Variance vs Samples (g=$g, J=$J, row=$row)",
        legend=:topright,
        linewidth=2,
        marker=:circle,
        markersize=4,
        yscale=:log10,
        xscale=:log10,
        size=(800, 600),
        grid=true,
        xticks=:auto,
        yticks=:auto
    )
    
    # Fit reference line: y = a/x
    # Transform to: y = a * (1/x)
    # Least squares without intercept: a = (Σ y_i/x_i) / (Σ 1/x_i²)
    X_inv = 1 ./ samples_list
    a = sum(var_list .* X_inv) / sum(X_inv .^ 2)
    @show a
    # Generate fitted line
    N_range = range(minimum(samples_list), maximum(samples_list), length=100)
    fitted_line = [a / N for N in N_range]
    
    Plots.plot!(p_plot, N_range, fitted_line,
               linestyle=:dash,
               linewidth=2,
               color=:red,
               label="y = $(round(a, digits=2))/x")
    
    # Save the figure
    if save_path === nothing
        save_path = "image/var_mean_samples_sys_g=$(g)_J=$(J)_row=$(row).pdf"
    end
    
    savefig(p_plot, save_path)
    @info "Figure saved to: $save_path"
    
    return p_plot, samples_list, var_list
end

function E_vs_qubits(g::Float64, J::Float64, row::Int, p::Int, nqubits::Int; data_dir="data", save_path=nothing,nshots=100)
    samples_list = Int[]
    all_energies = Float64[]
    all_samples = Int[]
    
    params_file = joinpath(data_dir, "compile_final_params_row=$(row)_g=$(g).dat")
    params = []
    open(params_file, "r") do file
        for line in eachline(file)
            if !startswith(strip(line), "#") && !isempty(strip(line))
                param_vec = parse.(Float64, split(line))
                push!(params, param_vec)
            end
        end
    end
    params = vcat(params...)

    for samples in 1000:1000:10000
        @info "Processing samples=$samples"
        energy_list = Float64[]
        for i in 1:nshots
            A_matrix = build_gate_from_params(params, p, row, nqubits)
            rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=1000, samples=samples, measure_first=:Z)
            energy = energy_measure(X_list, Z_list, g, J, row) 
            push!(energy_list, energy)
            push!(all_energies, energy)
            push!(all_samples, samples)
        end
        push!(samples_list, samples)
    end
    
    # Create scatter plot
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], 
              xlabel="1/Samples", 
              ylabel="Energy",
              title="Energy vs 1/Samples (g=$g, J=$J, row=$row, nshots=$nshots)")
    
    scatter!(ax, 1 ./ all_samples, all_energies, markersize=8, alpha=0.6)
    
    if save_path === nothing
        save_path = "image/E_vs_samples_sys_g=$(g)_J=$(J)_row=$(row).pdf"
    end
    
    savefig(p_plot, save_path)
    @info "Figure saved to: $save_path"
    
    return fig
end

