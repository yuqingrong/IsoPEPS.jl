"""
    plot_dimer_structure_factor(filename; nq=50, dimer_orientation=:vertical,
                                max_separation=20, use_exact=true,
                                conv_step=1000, samples=100000, save_path=nothing)

Brillouin-zone heatmap of the dimer static structure factor S_D(qx, qy).

# Arguments
- `filename`: Path to a saved optimization result JSON
- `nq`: Number of q-points along each axis (total grid: nq × nq)
- `dimer_orientation`: `:vertical` or `:horizontal`
- `max_separation`: Max column separation in the structure factor sum
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, SD)` where `SD` is the nq × nq matrix of S_D values
"""
function plot_dimer_structure_factor(filename::String;
                                     nq::Int=50,
                                     dimer_orientation::Symbol=:vertical,
                                     max_separation::Int=20,
                                     use_exact::Bool=true,
                                     conv_step::Int=1000,
                                     samples::Int=100000,
                                     save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    qvals = range(0.0, 2Float64(π), length=nq)
    SD = zeros(nq, nq)

    method_str = use_exact ? "exact" : "sampling"
    println("=== Dimer Structure Factor S_D(q) [$method_str, $dimer_orientation] ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, nq=$nq, max_sep=$max_separation")

    if use_exact
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end

        N_uc = length(op.columns)  # columns per unit cell (1 or 2)
        row = op.row
        vq  = op.virtual_qubits
        paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]

        T_cols = _column_transfer_matrices(op)
        T_combined = reduce(*, T_cols)
        l_vec, r_vec, nf, _ = _fixed_points(T_combined)

        # Build per-(col, pos) single-column dimer TMs and full-period dimer TMs
        println("Building dimer transfer matrices (N_uc=$N_uc)...")
        T_D_col_map  = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
        T_D_period_map = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
        μ_map = Dict{Tuple{Int,Int}, ComplexF64}()

        for κ in 1:N_uc, pos in 1:_row
            pos2 = pos % row + 1
            T_D_κ = zeros(ComplexF64, size(T_cols[1]))
            for σ in paulis
                E = get_transfer_matrix_with_operator(
                    op.columns[κ], row, vq, Dict(pos => σ, pos2 => σ);
                    optimizer=GreedyMethod())
                T_D_κ .+= E
            end
            T_D_κ ./= 4.0
            T_D_col_map[(κ, pos)] = T_D_κ

            T_before = κ > 1 ? reduce(*, T_cols[1:κ-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            T_after  = κ < N_uc ? reduce(*, T_cols[κ+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            T_D = T_before * T_D_κ * T_after
            T_D_period_map[(κ, pos)] = T_D
            μ_map[(κ, pos)] = dot(l_vec, T_D * r_vec) / nf
        end

        # Subtract global mean squared to preserve VBS signal
        # (bond-specific μ1*μ2 removes the alternation between unit cell columns)
        D_avg = real(Statistics.mean(collect(values(μ_map))))
        D_avg_sq = D_avg^2

        # Precompute correlations: corr_cache[(κ1,p1,κ2,p2)][m] = ⟨DD⟩ - D_avg²
        println("Precomputing dimer-dimer correlations...")
        corr_cache = Dict{NTuple{4,Int}, Dict{Int,ComplexF64}}()

        for κ1 in 1:N_uc, pos1 in 1:_row, κ2 in 1:N_uc, pos2 in 1:_row
            corrs = Dict{Int, ComplexF64}()
            pos1b = pos1 % row + 1
            pos2b = pos2 % row + 1

            # --- sep=0 (same period): build combined single-period TM ---
            if κ1 == κ2 && pos1 == pos2
                # ⟨D²⟩ at same bond: 4-operator insertion
                val0 = zero(ComplexF64)
                T_b = κ1 > 1 ? reduce(*, T_cols[1:κ1-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_a = κ1 < N_uc ? reduce(*, T_cols[κ1+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                for σa in paulis, σb in paulis
                    O_p  = σa * σb
                    O_p2 = σa * σb
                    E = get_transfer_matrix_with_operator(
                        op.columns[κ1], row, vq, Dict(pos1 => O_p, pos1b => O_p2);
                        optimizer=GreedyMethod())
                    val0 += dot(l_vec, T_b * E * T_a * r_vec) / nf
                end
                corrs[0] = val0 / 16.0 - D_avg_sq

            elseif κ1 == κ2
                # Same column, different positions: 4-operator single-column insertion
                val0 = zero(ComplexF64)
                T_b = κ1 > 1 ? reduce(*, T_cols[1:κ1-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_a = κ1 < N_uc ? reduce(*, T_cols[κ1+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                for σa in paulis, σb in paulis
                    ops = Dict{Int, Matrix{ComplexF64}}()
                    for (p, op_mat) in [(pos1, σa), (pos1b, σa), (pos2, σb), (pos2b, σb)]
                        if haskey(ops, p)
                            ops[p] = ops[p] * op_mat
                        else
                            ops[p] = copy(op_mat)
                        end
                    end
                    E = get_transfer_matrix_with_operator(
                        op.columns[κ1], row, vq, ops; optimizer=GreedyMethod())
                    val0 += dot(l_vec, T_b * E * T_a * r_vec) / nf
                end
                corrs[0] = val0 / 16.0 - D_avg_sq

            else
                # Different columns in same period: compose single-column dimer TMs
                lo, hi = minmax(κ1, κ2)
                T_b = lo > 1 ? reduce(*, T_cols[1:lo-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_mid = hi > lo + 1 ? reduce(*, T_cols[lo+1:hi-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_a = hi < N_uc ? reduce(*, T_cols[hi+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                if κ1 < κ2
                    T_DD = T_b * T_D_col_map[(κ1, pos1)] * T_mid * T_D_col_map[(κ2, pos2)] * T_a
                else
                    T_DD = T_b * T_D_col_map[(κ2, pos2)] * T_mid * T_D_col_map[(κ1, pos1)] * T_a
                end
                corrs[0] = dot(l_vec, T_DD * r_vec) / nf - D_avg_sq
            end

            # --- sep >= 1 (different periods) ---
            T_D1 = T_D_period_map[(κ1, pos1)]
            T_D2 = T_D_period_map[(κ2, pos2)]
            l_TD1 = T_D1' * l_vec
            max_period_sep = max(1, max_separation ÷ N_uc)
            for m in 1:max_period_sep
                current = T_D2 * r_vec
                for _ in 1:(m - 1)
                    current = T_combined * current
                end
                corrs[m] = dot(l_TD1, current) / nf - D_avg_sq
            end

            corr_cache[(κ1, pos1, κ2, pos2)] = corrs
            print("\r  (κ=$κ1,p=$pos1)→(κ=$κ2,p=$pos2)")
        end
        println("\nFourier transforming over $nq × $nq q-grid...")

        # Use max_separation as physical columns (not periods) for consistency with sampling
        max_period = max(1, max_separation ÷ N_uc)
        max_col_sep = max_period * N_uc + (N_uc - 1)
        L_eff = Float64(max_col_sep + 1)  # effective finite system length
        N_d = N_uc * _row
        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                val = 0.0
                for κ1 in 1:N_uc, pos1 in 1:_row, κ2 in 1:N_uc, pos2 in 1:_row
                    Δp = pos2 - pos1
                    corrs = corr_cache[(κ1, pos1, κ2, pos2)]
                    for (m, cval) in corrs
                        Δx = m * N_uc + (κ2 - κ1)
                        if Δx < 0
                            continue
                        end
                        # Bartlett (triangular) window: simulates finite-system weighting
                        w = 1.0 - abs(Δx) / L_eff
                        w <= 0.0 && continue
                        if Δx == 0
                            val += w * cos(qy * Δp) * real(cval)
                        else
                            val += 2.0 * w * cos(qx * Δx + qy * Δp) * real(cval)
                        end
                    end
                end
                SD[i, j] = val / N_d
            end
        end
    else
        resample_result = resample_circuit(filename; conv_step=conv_step,
                                            samples=samples, measure_y=true)
        isnothing(resample_result) && error("Resampling failed for $filename")
        _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = Y_samples[conv_step+1:end]

        # Precompute dimer values matrix once
        println("Precomputing dimer values from samples...")
        all_samples = (X_vec, Y_vec, Z_vec)
        ncols = length(Z_vec) ÷ _row

        if dimer_orientation == :vertical
            dimer_vals = zeros(_row, ncols)
            for S in all_samples
                for c in 1:ncols, pos in 1:_row
                    pos2 = pos % _row + 1
                    i1 = _row * (c - 1) + pos
                    i2 = _row * (c - 1) + pos2
                    dimer_vals[pos, c] += S[i1] * S[i2] / 4.0
                end
            end
            n_cols_d = ncols
        else  # :horizontal
            dimer_vals = zeros(_row, ncols - 1)
            for S in all_samples
                for c in 1:(ncols - 1), pos in 1:_row
                    i1 = _row * (c - 1) + pos
                    i2 = _row * c + pos
                    dimer_vals[pos, c] += S[i1] * S[i2] / 4.0
                end
            end
            n_cols_d = ncols - 1
        end

        max_sep = min(max_separation, n_cols_d - 1)
        n_pos = _row
        μ = vec(mean(dimer_vals, dims=2))

        # Precompute correlation tables
        println("Precomputing correlation tables...")
        # corr0[p1, p2] = mean over c of dimer_vals[p1,c]*dimer_vals[p2,c]
        corr0 = zeros(n_pos, n_pos)
        for p1 in 1:n_pos, p2 in 1:n_pos
            corr0[p1, p2] = mean(dimer_vals[p1, c] * dimer_vals[p2, c] for c in 1:n_cols_d)
        end
        # corr_dc[Δc][p1, p2]
        corr_dc = Vector{Matrix{Float64}}(undef, max_sep)
        for Δc in 1:max_sep
            m = zeros(n_pos, n_pos)
            for p1 in 1:n_pos, p2 in 1:n_pos
                m[p1, p2] = mean(dimer_vals[p1, c] * dimer_vals[p2, c + Δc] for c in 1:(n_cols_d - Δc))
            end
            corr_dc[Δc] = m
        end

        println("Fourier transforming over $nq × $nq q-grid...")
        μ_avg = mean(μ)
        μ_avg_sq = μ_avg^2
        L_eff_s = Float64(max_sep + 1)
        N_d = n_pos
        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                val = 0.0
                for p1 in 1:n_pos, p2 in 1:n_pos
                    Δp = p2 - p1
                    val += cos(qy * Δp) * (corr0[p1, p2] - μ_avg_sq)
                end
                for Δc in 1:max_sep
                    w = 1.0 - Δc / L_eff_s
                    for p1 in 1:n_pos, p2 in 1:n_pos
                        Δp = p2 - p1
                        val += 2.0 * w * cos(qx * Δc + qy * Δp) * (corr_dc[Δc][p1, p2] - μ_avg_sq)
                    end
                end
                SD[i, j] = val / N_d
            end
        end
    end

    # --- Plot ---
    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1],
              xlabel="qₓ",
              ylabel="qᵧ",
              title="Dimer Structure Factor S_D(q) [$dimer_orientation]",
              aspect=DataAspect(),
              xticks=([0, π, 2π], ["0", "π", "2π"]),
              yticks=([0, π, 2π], ["0", "π", "2π"]))

    hm = heatmap!(ax, qvals, qvals, SD, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_D(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, SD)
end

"""
    plot_spin_structure_factor(filename; nq=50, max_separation=20, use_exact=true,
                               conv_step=1000, samples=100000, save_path=nothing)

Brillouin-zone heatmap of the spin-spin static structure factor S_SS(qx, qy).

# Arguments
- `filename`: Path to a saved optimization result JSON
- `nq`: Number of q-points along each axis (total grid: nq × nq)
- `max_separation`: Max column separation in the structure factor sum
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, SSS)` where `SSS` is the nq × nq matrix of S_SS values
"""
function plot_spin_structure_factor(filename::String;
                                    nq::Int=50,
                                    max_separation::Int=10,
                                    use_exact::Bool=true,
                                    conv_step::Int=1000,
                                    samples::Int=100000,
                                    J2::Float64=0.0,
                                    D::Int=2,
                                    save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    qvals = range(0.0, 2Float64(π), length=nq)
    SSS = zeros(nq, nq)

    method_str = use_exact ? "exact" : "sampling"
    println("=== Spin Structure Factor S_SS(q) [$method_str] ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, nq=$nq, max_sep=$max_separation")

    if use_exact
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end

        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                SSS[i, j] = spin_spin_structure_factor(op, (qx, qy);
                                max_separation=max_separation)
            end
            print("\r  qx $i/$nq")
        end
        println()
    else
        resample_result = resample_circuit(filename; conv_step=conv_step,
                                            samples=samples, measure_y=true)
        isnothing(resample_result) && error("Resampling failed for $filename")
        _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = Y_samples[conv_step+1:end]

        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                SSS[i, j] = spin_spin_structure_factor(X_vec, Z_vec, Y_vec, _row, (qx, qy);
                                max_separation=max_separation)
            end
            print("\r  qx $i/$nq")
        end
        println()
    end

    # --- Plot ---
    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1],
              xlabel="qₓ",
              ylabel="qᵧ",
              title="S_SS(q)  J₂=$J2, D=$D",
              aspect=DataAspect(),
              xticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]),
              yticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]))

    hm = heatmap!(ax, qvals, qvals, SSS, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_SS(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, SSS)
end

"""
    save_combined_structure_factor_data(output_file, data_dir, J2_values; kwargs...)

Compute spin and dimer structure factor matrices for each J2 value and save the
results to a JSON file. The saved file can later be passed to
`plot_combined_structure_factors` via the `data_file` keyword to skip
recomputation.

# Arguments
- `output_file`: Path to write the JSON data file
- `data_dir`: Directory containing saved optimization result JSONs
- `J2_values`: Vector of J2 coupling values (e.g., [0.0, 0.5, 1.0])
- `J1`: J1 coupling (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters for filename matching
- `nq`: Number of q-points along each axis (grid: nq × nq)
- `max_separation_spin`: Max column separation for spin structure factor
- `max_separation_dimer`: Max column separation for dimer structure factor
- `dimer_orientation`: `:vertical` or `:horizontal`
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`, `samples`: Sampling parameters (when `use_exact=false`)

# Returns
- `(spin_matrices, dimer_matrices)` — Vectors of nq×nq matrices
"""
function save_combined_structure_factor_data(output_file::String,
        data_dir::String, J2_values::Vector{Float64};
        J1::Float64=1.0,
        row::Int=4, p::Int=3, nqubits::Int=3,
        nq::Int=50,
        max_separation_spin::Int=10,
        max_separation_dimer::Int=20,
        dimer_orientation::Symbol=:vertical,
        use_exact::Bool=true,
        conv_step::Int=1000,
        samples::Int=100000)

    n = length(J2_values)
    spin_matrices = Vector{Matrix{Float64}}(undef, n)
    dimer_matrices = Vector{Matrix{Float64}}(undef, n)
    filenames = Vector{String}(undef, n)

    for (idx, val) in enumerate(J2_values)
        candidates = [
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "exact_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
        ]
        found = ""
        for c in candidates
            if isfile(c)
                found = c
                break
            end
        end
        isempty(found) && error("No file found for J2=$val, tried $(length(candidates)) patterns")
        filenames[idx] = found
        println("J2=$val  →  $(basename(found))")
    end

    for idx in 1:n
        println("\n--- Computing spin structure factor for J2=$(J2_values[idx]) ---")
        _, SSS = plot_spin_structure_factor(filenames[idx];
                    nq=nq, max_separation=max_separation_spin,
                    use_exact=use_exact, conv_step=conv_step, samples=samples)
        spin_matrices[idx] = SSS

        println("\n--- Computing dimer structure factor for J2=$(J2_values[idx]) ---")
        _, SD = plot_dimer_structure_factor(filenames[idx];
                    nq=nq, dimer_orientation=dimer_orientation,
                    max_separation=max_separation_dimer,
                    use_exact=use_exact, conv_step=conv_step, samples=samples)
        dimer_matrices[idx] = SD
    end

    save_results(output_file;
        J2_values=J2_values,
        nq=nq,
        use_exact=use_exact,
        spin_matrices=[collect(eachcol(m)) for m in spin_matrices],
        dimer_matrices=[collect(eachcol(m)) for m in dimer_matrices])
    println("\nData saved to: $output_file")
    return (spin_matrices, dimer_matrices)
end

"""
    plot_combined_structure_factors(data_dir, J2_values; kwargs...)

Combined 2-row × N-column panel figure: spin structure factor S(q) on top,
dimer structure factor Sᴅ(q) on bottom, one column per J2 value, with shared
colorbars per row.

# Arguments
- `data_dir`: Directory containing saved optimization result JSONs
- `J2_values`: Vector of J2 coupling values (e.g., [0.0, 0.5, 1.0])
- `J1`: J1 coupling (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters for filename matching
- `nq`: Number of q-points along each axis (grid: nq × nq)
- `max_separation_spin`: Max column separation for spin structure factor
- `max_separation_dimer`: Max column separation for dimer structure factor
- `dimer_orientation`: `:vertical` or `:horizontal`
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`, `samples`: Sampling parameters (when `use_exact=false`)
- `data_file`: Optional path to a JSON produced by `save_combined_structure_factor_data`.
  When provided the matrices are loaded from disk and no computation is performed;
  `data_dir`, `J2_values`, and all method parameters are ignored.
- `figsize`: `(width, height)` in points. Defaults to double-column APS width (510 pt)
  with height derived from the number of columns so each heatmap stays roughly square.
- `save_path`: Optional path to save the figure

# Returns
- `(fig, spin_matrices, dimer_matrices)` where each `*_matrices` is a Vector of nq×nq matrices
"""
function plot_combined_structure_factors(data_dir::String, J2_values::Vector{Float64};
        J1::Float64=1.0,
        row::Int=4, p::Int=3, nqubits::Int=3,
        nq::Int=50,
        max_separation_spin::Int=10,
        max_separation_dimer::Int=20,
        dimer_orientation::Symbol=:vertical,
        use_exact::Bool=true,
        conv_step::Int=1000,
        samples::Int=100000,
        data_file=nothing,
        figsize=nothing,
        save_path=nothing)

    local spin_matrices, dimer_matrices

    if !isnothing(data_file)
        # --- Load pre-computed data ---
        println("Loading structure factor data from: $data_file")
        d = load_results(data_file)
        J2_values = Float64.(d["J2_values"])
        nq = Int(d["nq"])
        spin_matrices  = [Float64.(hcat(col...)) for col in d["spin_matrices"]]
        dimer_matrices = [Float64.(hcat(col...)) for col in d["dimer_matrices"]]
    else
        spin_matrices = Vector{Matrix{Float64}}(undef, length(J2_values))
        dimer_matrices = Vector{Matrix{Float64}}(undef, length(J2_values))
        filenames = Vector{String}(undef, length(J2_values))

        # --- Find files ---
        for (idx, val) in enumerate(J2_values)
            candidates = [
                joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
                joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
                joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
                joinpath(data_dir, "exact_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
            ]
            found = ""
            for c in candidates
                if isfile(c)
                    found = c
                    break
                end
            end
            if isempty(found)
                error("No file found for J2=$val, tried $(length(candidates)) patterns")
            end
            filenames[idx] = found
            println("J2=$val  →  $(basename(found))")
        end

        # --- Compute structure factor matrices ---
        for idx in 1:length(J2_values)
            println("\n--- Computing spin structure factor for J2=$(J2_values[idx]) ---")
            _, SSS = plot_spin_structure_factor(filenames[idx];
                        nq=nq, max_separation=max_separation_spin,
                        use_exact=use_exact, conv_step=conv_step, samples=samples)
            spin_matrices[idx] = SSS

            println("\n--- Computing dimer structure factor for J2=$(J2_values[idx]) ---")
            _, SD = plot_dimer_structure_factor(filenames[idx];
                        nq=nq, dimer_orientation=dimer_orientation,
                        max_separation=max_separation_dimer,
                        use_exact=use_exact, conv_step=conv_step, samples=samples)
            dimer_matrices[idx] = SD
        end
    end

    # --- Shared color ranges ---
    spin_min = minimum(minimum.(spin_matrices))
    spin_max = maximum(maximum.(spin_matrices))
    dimer_min = minimum(minimum.(dimer_matrices))
    dimer_max = maximum(maximum.(dimer_matrices))

    qvals = range(0.0, 2Float64(π), length=nq)

    # --- Build combined figure ---
    # Default: double-column APS width; height derived so each heatmap is square.
    # panel_width ≈ (total_width - colorbar_col) / n_columns
    # total_height ≈ n_rows × panel_width + top/bottom margins
    n = length(J2_values)
    _colorbar_w = 50
    _default_w  = first(PAPER_FIGSIZE_WIDE)   # 510 pt
    _panel_w    = (_default_w - _colorbar_w) ÷ n
    _default_h  = 2 * _panel_w + 60           # 2 rows + label margin
    _figsize    = isnothing(figsize) ? (_default_w, _default_h) : figsize
    fig = Figure(size=_figsize)

    local hm_spin, hm_dimer

    # Top row: spin structure factor S(q)
    for (j, J2) in enumerate(J2_values)
        ax = Axis(fig[1, j],
                  aspect=DataAspect(),
                  title="J₂ = $J2",
                  xticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]),
                  yticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]))
        if j == 1
            ax.ylabel = "qᵧ"
        else
            ax.yticklabelsvisible = false
        end
        ax.xticklabelsvisible = false
        hm_spin = heatmap!(ax, qvals, qvals, spin_matrices[j],
                           colormap=:viridis, colorrange=(spin_min, spin_max))
    end
    Colorbar(fig[1, n + 1], hm_spin, label="S(q)")

    # Bottom row: dimer structure factor Sᴅ(q)
    for (j, J2) in enumerate(J2_values)
        ax = Axis(fig[2, j],
                  xlabel="qₓ",
                  aspect=DataAspect(),
                  xticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]),
                  yticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]))
        if j == 1
            ax.ylabel = "qᵧ"
        else
            ax.yticklabelsvisible = false
        end
        hm_dimer = heatmap!(ax, qvals, qvals, dimer_matrices[j],
                            colormap=:viridis, colorrange=(dimer_min, dimer_max))
    end
    Colorbar(fig[2, n + 1], hm_dimer, label="Sᴅ(q)")

    # Row labels on the left
    Label(fig[1, 0], "S(q)", rotation=π/2, fontsize=16, tellheight=false)
    Label(fig[2, 0], "Sᴅ(q)", rotation=π/2, fontsize=16, tellheight=false)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, spin_matrices, dimer_matrices)
end

"""
    plot_bond_energy_pattern(filename; max_cols=10, use_exact=true,
                             conv_step=1000, samples=100000, title="",
                             save_path=nothing)

Visualize bond energies ⟨S_i · S_j⟩ on every nearest-neighbour bond of the
cylinder lattice. Strong/weak bond alternation directly reveals VBS order.

# Arguments
- `filename`: Path to a saved optimization result JSON
- `max_cols`: Number of columns to display
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `title`: Optional figure title
- `save_path`: Optional path to save the figure

# Returns
- `(fig, bond_data)` where `bond_data` is a Dict with keys
  `:vertical => Matrix{Float64}(row, max_cols)` and
  `:horizontal => Matrix{Float64}(row, max_cols-1)`
"""
function plot_bond_energy_pattern(filename::String;
                                  max_cols::Int=10,
                                  use_exact::Bool=true,
                                  conv_step::Int=1000,
                                  samples::Int=100000,
                                  figsize=nothing,
                                  save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    method_str = use_exact ? "exact" : "sampling"
    println("=== Bond Energy Pattern [$method_str] ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, max_cols=$max_cols")

    # Unit-cell bond expectations: vert_uc[pos, col], horiz_uc[pos, col]
    if use_exact
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end
        vert_uc, horiz_uc = all_bond_expectations(op)
        N_uc = length(op.columns)
        println("  Unit cell columns: $N_uc")
        println("  Vertical bond expectations:")
        for pos in 1:_row, c in 1:N_uc
            pos2 = pos % _row + 1
            println("    ($pos↔$pos2, col=$c): $(vert_uc[pos, c])")
        end
        if size(horiz_uc, 2) > 0
            println("  Horizontal bond expectations (intra-cell):")
            for pos in 1:_row, c in 1:size(horiz_uc, 2)
                println("    (pos=$pos, col=$c↔$(c+1)): $(horiz_uc[pos, c])")
            end
        end
        # Inter-period horizontal bond = horizontal bond spanning last col of period to first of next
        # This is the same as bond_expectation with orientation=:horizontal across period boundary
        # For 1x1 UC, ALL horizontal bonds are inter-period (computed via transfer matrix)
        # We compute it via: ⟨σ^α_{N,pos}(period k) σ^α_{1,pos}(period k+1)⟩ / 4
        # This requires the correlation function approach
        horiz_inter = zeros(Float64, _row)
        for pos in 1:_row
            corr = spin_spin_correlation(op, [1]; col1=N_uc, col2=1, pos1=pos, pos2=pos)
            horiz_inter[pos] = real(corr[1])
        end
        println("  Inter-period horizontal bond expectations:")
        for pos in 1:_row
            println("    (pos=$pos, across period): $(horiz_inter[pos])")
        end

    else
        # Sampling branch
        resample_result = resample_circuit(filename; conv_step=conv_step, samples=samples)
        isnothing(resample_result) && error("Resampling failed for $filename")
        if length(resample_result) == 6
            _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        else
            _rho, Z_samples, X_samples, _params, _gates = resample_result
            Y_samples = zeros(length(X_samples))
        end
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = length(resample_result) == 6 ? Y_samples[conv_step+1:end] : zeros(length(Z_vec))

        dimer_vals_v, dimer_vals_h = _build_all_dimer_values(X_vec, Z_vec, Y_vec, _row)

        # Average over columns to get unit-cell pattern
        N_uc = is_2x2 ? 2 : 1
        ncols_v = size(dimer_vals_v, 2)
        ncols_h = size(dimer_vals_h, 2)

        vert_uc = zeros(Float64, _row, N_uc)
        for pos in 1:_row, c in 1:N_uc
            cols = c:N_uc:ncols_v
            vert_uc[pos, c] = mean(dimer_vals_v[pos, cols])
        end

        # Horizontal bonds: for 1x1 UC, all horizontal bonds are equivalent
        # For 2x2 UC, separate intra-cell and inter-cell
        if N_uc == 1
            horiz_uc = zeros(Float64, _row, 0)
            horiz_inter = zeros(Float64, _row)
            for pos in 1:_row
                horiz_inter[pos] = mean(dimer_vals_h[pos, :])
            end
        else
            horiz_uc = zeros(Float64, _row, N_uc - 1)
            for pos in 1:_row
                # Intra-cell: odd-indexed horizontal bonds (col 1→2)
                intra_cols = 1:2:ncols_h
                horiz_uc[pos, 1] = mean(dimer_vals_h[pos, intra_cols])
            end
            horiz_inter = zeros(Float64, _row)
            for pos in 1:_row
                # Inter-cell: even-indexed horizontal bonds (col 2→3, i.e., period boundary)
                inter_cols = 2:2:ncols_h
                horiz_inter[pos] = mean(dimer_vals_h[pos, inter_cols])
            end
        end

        println("  Vertical bond expectations (column-averaged):")
        for pos in 1:_row, c in 1:N_uc
            pos2 = pos % _row + 1
            println("    ($pos↔$pos2, col=$c): $(vert_uc[pos, c])")
        end
        if size(horiz_uc, 2) > 0
            println("  Horizontal bond expectations (intra-cell, averaged):")
            for pos in 1:_row
                println("    (pos=$pos): $(horiz_uc[pos, 1])")
            end
        end
        println("  Inter-period horizontal bond expectations (averaged):")
        for pos in 1:_row
            println("    (pos=$pos): $(horiz_inter[pos])")
        end
    end

    # Tile unit-cell pattern over max_cols columns
    vert_tiled = zeros(Float64, _row, max_cols)
    for col in 1:max_cols, pos in 1:_row
        c_uc = ((col - 1) % N_uc) + 1
        vert_tiled[pos, col] = vert_uc[pos, c_uc]
    end

    horiz_tiled = zeros(Float64, _row, max_cols - 1)
    for col in 1:(max_cols - 1), pos in 1:_row
        c_uc = ((col - 1) % N_uc) + 1
        c_next_uc = (col % N_uc) + 1
        if c_next_uc > c_uc && size(horiz_uc, 2) >= c_uc
            # Intra-cell horizontal bond
            horiz_tiled[pos, col] = horiz_uc[pos, c_uc]
        else
            # Inter-period horizontal bond
            horiz_tiled[pos, col] = horiz_inter[pos]
        end
    end

    # --- Drawing ---
    all_vals = vcat(vec(vert_tiled), vec(horiz_tiled))
    cmax = isempty(all_vals) ? 1.0 : max(maximum(abs, all_vals), 1e-10)

    # Figure size: scale one lattice unit to ~35 pt, add colorbar column and margins
    _unit    = 35
    _cb_w    = 55
    _default_w = max_cols * _unit + _cb_w + 20
    _default_h = (_row + 1) * _unit + 20
    _figsize = isnothing(figsize) ? (_default_w, _default_h) : figsize

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax = Axis(fig[1, 1];
                  xlabel  = "Column",
                  ylabel  = "Row",
                  aspect  = DataAspect(),
                  xticklabelsize = PAPER_TICKLABELSIZE,
                  yticklabelsize = PAPER_TICKLABELSIZE)

        # Vertical bonds
        for col in 1:max_cols
            for pos in 1:_row
                pos2 = pos % _row + 1
                val  = vert_tiled[pos, col]
                y1, y2 = Float64(pos), Float64(pos2)
                lw = 0.8 + 3.0 * abs(val) / cmax
                c  = val / cmax
                if pos2 < pos
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y1 + 0.5],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                    linesegments!(ax, [Float64(col), Float64(col)], [y2 - 0.5, y2],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                else
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y2],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                end
            end
        end

        # Horizontal bonds
        for col in 1:(max_cols - 1)
            for pos in 1:_row
                val = horiz_tiled[pos, col]
                lw  = 0.8 + 3.0 * abs(val) / cmax
                c   = val / cmax
                linesegments!(ax, [Float64(col), Float64(col + 1)],
                              [Float64(pos), Float64(pos)],
                              color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                              linewidth=lw)
            end
        end

        # Sites
        xs = [Float64(col) for col in 1:max_cols for _ in 1:_row]
        ys = [Float64(pos) for _   in 1:max_cols for pos in 1:_row]
        scatter!(ax, xs, ys; color=:gray30, markersize=5, strokewidth=0)

        Colorbar(fig[1, 2]; colormap=:RdBu, limits=(-cmax, cmax),
                 label="⟨𝐒ᵢ · 𝐒ⱼ⟩",
                 labelsize=PAPER_AXIS_LABELSIZE,
                 ticklabelsize=PAPER_TICKLABELSIZE,
                 width=12)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    bond_data = Dict(:vertical => vert_tiled, :horizontal => horiz_tiled)
    return (fig, bond_data)
end
