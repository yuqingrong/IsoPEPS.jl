function _save_training_data(g::Float64, row::Int, energy_history, params_history, Z_list_list, X_list_list, gap_list, eigenvalues_list, final_params, final_cost, gap_final; data_dir="data", measure_first=:X)
    if !isdir(data_dir)
        mkdir(data_dir)
    end
    # Save energy history
    open(joinpath(data_dir, "compile_energy_history_row=$(row)_g=$(g)_.dat"), "w") do io
        for energy in energy_history
            println(io, energy)
        end
    end
    # Save params history (each row is one parameter set)
    open(joinpath(data_dir, "compile_params_history_row=$(row)_g=$(g).dat"), "w") do io
        for params in params_history
            println(io, join(params, " "))
        end
    end
    # Save Z_list_list (each row is one Z_list)
    open(joinpath(data_dir, "compile_Z_list_list_row=$(row)_g=$(g).dat"), "w") do io
        for Z_list in Z_list_list
            println(io, join(Z_list, " "))
        end
    end

    # Save X_list_list (each row is one X_list)
    open(joinpath(data_dir, "compile_X_list_list_row=$(row)_g=$(g).dat"), "w") do io
        for X_list in X_list_list
            println(io, join(X_list, " "))
        end
    end
    # Save gap list
    open(joinpath(data_dir, "compile_gap_list_row=$(row)_g=$(g).dat"), "w") do io
        for gap in gap_list
            println(io, gap)
        end
    end
    # Save eigenvalues list
    open(joinpath(data_dir, "compile_eigenvalues_list_row=$(row)_g=$(g).dat"), "w") do io
        for eigenvalues in eigenvalues_list
            println(io, join(eigenvalues, " "))
        end
    end
    
    open(joinpath(data_dir, "compile_final_params_row=$(row)_g=$(g).dat"), "w") do io
        for params in final_params
            println(io, join(params, " "))
        end
    end

    open(joinpath(data_dir, "compile_final_cost_row=$(row)_g=$(g).dat"), "w") do io
        for cost in final_cost
            println(io, cost)
        end
    end

    open(joinpath(data_dir, "compile_gap_final_row=$(row)_g=$(g).dat"), "w") do io
        for gap in gap_final
            println(io, gap)
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

function ACF(g::Float64; measure_first=:X, data_dir="data", save_path=nothing, max_lag=nothing)
    
    filename = joinpath(data_dir, "$(measure_first)_first_$(measure_first)_list_list_g=$(g).dat")
    
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
        
        # Parse the Z values from the last line
               # Parse the Z values from the last line (excluding indices that are multiples of 3)
        #O = [parse(Float64, x) for (i, x) in enumerate(split(last_line)) if i % 3 != 0]
        #O = [parse(Float64, x) for (i, x) in enumerate(split(last_line)) if i % 4 == 0]
        O = parse.(Float64, split(last_line))
        N = length(O)
        @info "g=$g: Found $N observable values in last line"
        
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
        C0 = sum(O_centered[i]^2 for i in 1:N)  # Variance using every other point
        
        # Get autocorrelation from BinningAnalysis for comparison
        LB = BinningAnalysis.LogBinner(O)
        acf_BA = BinningAnalysis.all_autocorrelations(LB)
        @info "BinningAnalysis autocorrelations (first 10): $(acf_BA[1:min(10, length(acf_BA))])"
        tau_BA = BinningAnalysis.tau(LB)
        @info "BinningAnalysis integrated autocorrelation time: τ = $tau_BA"
        
        # Calculate autocorrelation function manually
        acf = zeros(max_lag)
        
        for k in 1:max_lag
            lag = k - 1
            
          
            # Calculate C(lag) = mean((O[i] - mean) * (O[i+lag] - mean))
            n_pairs = N - lag
            sum_prod = 0.0
            for i in 1:n_pairs
                sum_prod += O_centered[i] * O_centered[i + lag]
                #sum_prod += O[i] * O[i + lag]
            end
            acf[k] = abs(sum_prod / C0)
        end
        
        @info "ACF calculated. acf[1] = $(acf[1]) (should be 1.0)"
        @info "First few ACF values: $(acf[1:min(10, length(acf))])"
        
        # Fit exponential model: exp(-x / ξ)
        model(x, p) = exp.(-x ./ p[1])  # p = [ξ]
        
        # Use data from lag=1 onwards for fitting (skip lag=0 which is always 1.0)
        lags_for_fit = collect(1:(max_lag-1))
        acf_for_fit = acf[2:end]  # Skip first point (lag=0)
        
        # Initial guess: A ≈ acf[2], ξ ≈ 1/log(acf[2]/acf[1])
        #A_init = acf[2]
        xi_init = 1.0 / (-log(abs(acf[2])))
        xi_init = max(0.1, min(xi_init, 100.0))  # Bound initial guess
        p0 = [xi_init]
        
        # Fit the model
        fit_result = nothing
        A_fit = NaN
        xi_fit = NaN
        try
            fit_result = curve_fit(model, lags_for_fit, acf_for_fit, p0)
            fitted_params = coef(fit_result)
            xi_fit = fitted_params[1]
            
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
        p = Plots.plot(
            lags,
            acf,
            xlabel="Lag k",
            ylabel="Autocorrelation C(k)",
            title="$(measure_first) Autocorrelation for g=$g, τ_c_fit =$(round(xi_fit, digits=2))",
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
            xi_fit = fitted_params[1]
            fitted_curve = model(lags, fitted_params)
            Plots.plot!(p, lags, fitted_curve, 
                linewidth=2, 
                linestyle=:dash, 
                color=:red,
                label="Fit: exp(-k/$(round(xi_fit, digits=2))); Binnng: τ = $tau_BA"
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

function correlation(g::Float64; measure_first=:X, data_dir="data", save_path=nothing, max_lag=nothing)
    
    filename = joinpath(data_dir, "$(measure_first)_first_X_list_list_g=$(g).dat")
    
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
    g_list = [0.01, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    MPSKit_list = [1/0.09938670503395748, 1/0.2767186242180228, 1/0.4558293623207221, 1/0.7611451195884831, 1/1.7296475381670806, 1/2.0838447345246576, 1/1.4864869208462, 1/1.2008972020030175, 1/1.0310843874420765]
    PEPSKitlist = [-1.999999610358945, -2.007814504995826, -2.0312864807194675, 
                   -2.0705176991971634, -2.1256518211812336, -2.1969439738505, 
                   -2.2846818634108392, -2.389277196571146, -2.511299175269]
    nocompile_list = [10.0617079483451, 3.6138131708760275, 2.193803922262251, 1.3138101013460692, 0.5781525165137043, 0.4798822256011485, 0.6727270481199901, 0.8327107524053712, 0.9698526492947356]
    contract_list = [-1.9999998917735167, -2.0078191059955737, -2.0312516166482886, 
                     -2.0703162733046936, -2.1268495602624498, -2.197127459624827, 
                     -2.287280846863297, -2.3905645948990832,  -2.505084442515908]
    compile_list = [10.840122867995094, 3.4380187943307026, 2.2297162969555773, 1.436068361228758, 0.5780242148899613, 0.47981080364804873, 0.6726520227106328, 0.8325439670588666, 0.9701494798268981]
    
    fig = Plots.plot(xlabel="g", ylabel="spectral gap: -ln |λ_1|)", 
                     title="spectral gap vs Transverse Field (1D)", 
                     ylims=(0.0,11.0), xlims=(0.00, 2.25), 
                     yscale=:linear, 
                     yticks=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                     xticks=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25], size=(800, 600))

    colors = [RGBA(1,0,0,0.5), RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), 
              RGBA(0,0,1,0.5), RGBA(0,0,0,0.5)]
    markers = [:+, :x, :star, :diamond, :circle]
    
    Plots.plot!(fig, g_list, MPSKit_list, label="MPSKit",
              color=colors[1], marker=markers[1], markersize=4, linewidth=2)
    #Plots.plot!(fig, g_list, PEPSKitlist, label="PEPSKit",
                #color=colors[2], marker=markers[2], markersize=4, linewidth=2)
    #Plots.plot!(fig, g_list, contract_list, label="contract_directly",
                  #color=colors[3], marker=markers[3], markersize=4, linewidth=2)
    
    Plots.plot!(fig, g_list, nocompile_list, label="nocompile",
               color=colors[5], marker=markers[5], markersize=4, linewidth=2)
    Plots.plot!(fig, g_list, compile_list, label="compile",
               color=colors[4], marker=markers[4], markersize=4, linewidth=2)

    Plots.plot!(fig, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    Plots.plot!(fig, legend=:topright, legendfontsize=10)
    Plots.savefig(fig, "spectral gap_vs_g4.pdf")
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

function energy_converge(g_values::Vector{Float64}; data_dir="data_exact")
    # Initialize plot
    p = Plots.plot(xlabel="Iteration", ylabel="Energy", 
                   title="Energy Convergence to  vs Iteration",
                   legend=:best,
                   size=(1000, 600),
                   grid=true,
                   gridwidth=1,
                   gridcolor=:gray,
                   gridalpha=0.3)
    
    # Read and plot data for each g value
    for g in g_values
        filename = joinpath(data_dir, "nocompile_energy_history_g=$(g)_.dat")
        
        if !isfile(filename)
            @warn "File not found: $filename, skipping g=$g"
            continue
        end
        
        # Read energy history
        energy_history = Float64[]
        open(filename, "r") do io
            for line in eachline(io)
                push!(energy_history, parse(Float64, strip(line)))
            end
        end
        
        # Plot energy vs iteration
        iterations = 1:length(energy_history)
        Plots.plot!(p, iterations, energy_history, 
                   label="g = $g",
                   linewidth=2,
                   alpha=0.8)
    end
    
  
    Plots.savefig(p, "image/energyconverge_g=$(g_values).pdf")

    
    # Display and return plot
    Plots.display(p)
    @info "Energy convergence plot generated for g values: $g_values"
    
    return p
end

function energy_con(g::Union{Float64, Vector{Float64}}; data_dir="data")
    # Convert single value to vector for uniform processing
    g_values = g isa Float64 ? [g] : g
    
    if isempty(g_values)
        @error "No g values provided"
        return nothing
    end
    
    # Initialize plot
    p = nothing
    colors = palette(:tab10)
    valid_plots = 0
    
    for (idx, g_val) in enumerate(g_values)
        # Construct filename
        filename = joinpath(data_dir, "X_first_energy_history_g=$(g_val)_.dat")
        
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
        @info "Found $n_iterations energy values for g=$(g_val)"
        
        # Create or add to plot
        if p === nothing
            # First valid plot - initialize
            p = Plots.plot(
                1:n_iterations,
                energy_data,
                xlabel="Iteration",
                ylabel="Energy",
                title="Energy Convergence",
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
        
        valid_plots += 1
    end
    
    if p === nothing
        @error "No valid data found for any g value"
        return nothing
    end
    
    # Save the plot
    if length(g_values) == 1
        save_path = "image/energy_convergence_g=$(g_values[1]).pdf"
    else
        g_str = join(g_values, "_")
        save_path = "image/energy_convergence_g=$(g_str).pdf"
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

function var_mean_samples(g::Float64, J::Float64, row::Int, p::Int, nqubits::Int; data_dir="data", save_path=nothing)
    var_list = Float64[]
    samples_list = Int[]
    
    params_file = joinpath(data_dir, "Z_first_3_params_history_g=$(g).dat")
    params = parse.(Float64, split(readlines(params_file)[end]))
    
    @info "Computing variance vs samples for g=$g, J=$J, row=$row"
    _, exact_energy = exact_E_from_params(g, J, p, row, nqubits; data_dir="data", optimizer=GreedyMethod())
    for samples in 1000:1000:10000
        @info "Processing samples=$samples"
        energy_list = Float64[]
        for i in 1:20
            A_matrix = build_gate_from_params(params, p, row, nqubits)
            rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row; conv_step=1000, samples=samples, measure_first=:Z)
            energy = energy_measure(X_list, Z_list, g, J, row) 
            push!(energy_list, energy)
        end
        @show energy_list
        #variance = var(energy_list)
        variance = mean((energy_list .-(exact_energy)) .^2)
        push!(var_list, variance)
        push!(samples_list, samples)
        @info "samples=$samples, variance=$variance"
    end
    
    # Create the plot
    p_plot = Plots.plot(
        samples_list,
        var_list,
        xlabel="Number of Samples",
        ylabel="Variance",
        title="Variance vs Samples (g=$g, J=$J, row=$row)",
        legend=false,
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
    
    # Add 1/N reference line
    N_range = [minimum(samples_list), maximum(samples_list)]
    reference_scale = var_list[1] * samples_list[1]
    reference_line = [reference_scale / N for N in N_range]
    Plots.plot!(p_plot, N_range, reference_line,
               linestyle=:dash,
               linewidth=2,
               color=:red,
               label="1/N scaling")
    
    # Save the figure
    if save_path === nothing
        save_path = "image/var_mean_samples_sys_g=$(g)_J=$(J)_row=$(row).pdf"
    end
    
    savefig(p_plot, save_path)
    @info "Figure saved to: $save_path"
    
    return p_plot, samples_list, var_list
end
