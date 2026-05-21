# =============================================================================
# DMRG Structure Factor Heatmaps
# =============================================================================

"""
    plot_dmrg_spin_structure_factor(result; nq=50, bulk_fraction=0.5, save_path=nothing)

Brillouin-zone heatmap of spin structure factor S_SS(q) from DMRG ground state.

`result` is the NamedTuple returned by `dmrg_ground_state_2d`.
"""
function plot_dmrg_spin_structure_factor(result;
                                         nq::Int=50,
                                         max_separation::Int=10,
                                         J2::Float64=0.0,
                                         D::Int=2,
                                         save_path=nothing)
    qvals = range(0.0, 2Float64(π), length=nq)
    SSS = zeros(nq, nq)

    Lx = result.Lx
    Ly = result.Ly
    N = Lx * Ly

    # Use center 2*max_sep+1 columns as reference region
    n_bulk_cols = 2 * max_separation + 1
    center = div(Lx, 2)
    col_lo = center - max_separation
    col_hi = center + max_separation
    col_lo = max(1, col_lo)
    col_hi = min(Lx, col_hi)

    println("=== DMRG Spin Structure Factor (Lx=$Lx, Ly=$Ly, max_sep=$max_separation) ===")
    println("  Bulk columns: $col_lo to $col_hi ($(col_hi - col_lo + 1) cols)")

    # Precompute correlation matrix ONCE (the expensive part)
    println("  Computing correlation matrices...")
    SdotS = compute_SdotS_matrix(result)

    # site_to_2d: 1D index -> (col, row)
    coords = Vector{Tuple{Int,Int}}(undef, N)
    for s in 1:N
        coords[s] = (div(s - 1, Ly) + 1, mod(s - 1, Ly) + 1)
    end

    bulk_sites = [s for s in 1:N if col_lo <= coords[s][1] <= col_hi]

    # Precompute dx, dy arrays for bulk site pairs, filtered by max_separation
    pair_dx = Float64[]
    pair_dy = Float64[]
    pair_SS = Float64[]
    n_ref = 0  # count reference sites (for normalization)
    for (a, s1) in enumerate(bulk_sites)
        i1, j1 = coords[s1]
        n_ref += 1
        for (b, s2) in enumerate(bulk_sites)
            i2, j2 = coords[s2]
            dx = i2 - i1
            abs(dx) > max_separation && continue
            push!(pair_dx, dx)
            push!(pair_dy, j2 - j1)
            push!(pair_SS, SdotS[s1, s2])
        end
    end

    # Fourier transform: S(q) = (1/N_ref) Σ_{pairs} ⟨Si·Sj⟩ cos(q·Δr)
    println("  Computing Fourier transform on $nq×$nq grid...")
    for (i, qx) in enumerate(qvals)
        for (j, qy) in enumerate(qvals)
            SSS[i, j] = sum(pair_SS .* cos.(qx .* pair_dx .+ qy .* pair_dy)) / n_ref
        end
    end

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], xlabel="qₓ", ylabel="qᵧ",
              title="DMRG S_SS(q)  J₂=$J2, D=$D",
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
    plot_dmrg_dimer_structure_factor(result; nq=50, bulk_cols=20,
                                     dimer_orientation=:vertical, save_path=nothing)

Brillouin-zone heatmap of connected dimer structure factor S_D(q) from DMRG ground state.

Precomputes the ⟨D_b D_b'⟩ matrix once (expensive), then Fourier transforms for each q.
"""
function plot_dmrg_dimer_structure_factor(result;
                                          nq::Int=50,
                                          bulk_cols::Int=20,
                                          dimer_orientation::Symbol=:vertical,
                                          save_path=nothing)
    Lx = result.Lx
    Ly = result.Ly
    psi = result.psi
    sites = result.sites

    println("=== DMRG Dimer Structure Factor (Lx=$Lx, Ly=$Ly, $dimer_orientation) ===")

    # Precompute bonds and correlation data
    col_lo = max(1, div(Lx - bulk_cols, 2) + 1)
    col_hi = min(Lx, col_lo + bulk_cols - 1)

    # Use the DMRG correlation function to get precomputed data
    corr_data = compute_dimer_dimer_correlation_dmrg(result;
                    dimer_orientation=dimer_orientation, bulk_cols=bulk_cols)
    bonds = corr_data.bonds
    D_exp = corr_data.dimer_expectations
    DD = corr_data.DD_matrix
    n = length(bonds)

    # Precompute bond center coordinates
    bond_coords = Vector{Tuple{Float64,Float64}}(undef, n)
    for bi in 1:n
        _, _, col, row = bonds[bi]
        if dimer_orientation == :vertical
            bond_coords[bi] = (Float64(col), Float64(row) + 0.5)
        else
            bond_coords[bi] = (Float64(col) + 0.5, Float64(row))
        end
    end

    # Subtract global mean squared (not bond-specific product) to preserve VBS signal
    # On finite DMRG cylinders, ⟨D_b⟩ itself alternates due to pinned VBS order;
    # subtracting D_exp * D_exp' would remove that signal entirely.
    D_avg = Statistics.mean(D_exp)
    C_conn = DD .- D_avg^2

    println("Fourier transforming over $nq × $nq q-grid...")
    qvals = range(0.0, 2Float64(π), length=nq)
    SD = zeros(nq, nq)

    for (i, qx) in enumerate(qvals)
        for (j, qy) in enumerate(qvals)
            val = 0.0
            for bi in 1:n, bj in 1:n
                dx = bond_coords[bj][1] - bond_coords[bi][1]
                dy = bond_coords[bj][2] - bond_coords[bi][2]
                val += C_conn[bi, bj] * cos(qx * dx + qy * dy)
            end
            SD[i, j] = val / n
        end
        print("\r  qx $i/$nq")
    end
    println()

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], xlabel="qₓ", ylabel="qᵧ",
              title="DMRG Dimer Structure Factor S_D(q) [$dimer_orientation]",
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
    plot_dmrg_bond_energy_pattern(result; bulk_cols=20, title="", save_path=nothing)

Visualize bond energies ⟨S_i · S_j⟩ on every nearest-neighbour bond of the
DMRG ground state on an open cylinder. Strong/weak bond alternation directly
reveals VBS order.

# Arguments
- `result`: DMRG result (from `dmrg_ground_state_2d`)
- `bulk_cols`: Number of bulk columns to display (centered, avoiding boundaries)
- `title`: Custom figure title
- `save_path`: Optional path to save the figure

# Returns
- `(fig, bond_data)` where `bond_data` is a NamedTuple with fields
  `SdotS_matrix`, `bonds_v`, `bonds_h`, `D_v`, `D_h`
"""
function plot_dmrg_bond_energy_pattern(result;
                                       bulk_cols::Int=20,
                                       title::String="",
                                       save_path=nothing)
    Lx = result.Lx
    Ly = result.Ly

    # Get full S·S matrix
    SdotS = compute_SdotS_matrix(result)

    # Determine bulk region
    bulk_cols = min(bulk_cols, Lx)
    margin = div(Lx - bulk_cols, 2)
    col_lo = max(1, margin + 1)
    col_hi = min(Lx, col_lo + bulk_cols - 1)

    # Build bond lists and extract ⟨S_i · S_j⟩
    # Vertical bonds: (col, row) ↔ (col, row%Ly+1)
    bonds_v = Tuple{Int,Int,Int,Int}[]  # (si, sj, col, row)
    for col in col_lo:col_hi, row in 1:Ly
        row2 = row % Ly + 1
        si = (col - 1) * Ly + row
        sj = (col - 1) * Ly + row2
        push!(bonds_v, (si, sj, col, row))
    end

    # Horizontal bonds: (col, row) ↔ (col+1, row)
    bonds_h = Tuple{Int,Int,Int,Int}[]
    for col in col_lo:(col_hi - 1), row in 1:Ly
        si = (col - 1) * Ly + row
        sj = col * Ly + row
        push!(bonds_h, (si, sj, col, row))
    end

    D_v = Float64[SdotS[si, sj] for (si, sj, _, _) in bonds_v]
    D_h = Float64[SdotS[si, sj] for (si, sj, _, _) in bonds_h]

    all_vals = vcat(D_v, D_h)
    cmax = isempty(all_vals) ? 1.0 : max(maximum(abs, all_vals), 1e-10)

    println("=== DMRG Bond Energy Pattern ===")
    println("Lx=$Lx, Ly=$Ly, bulk cols=$col_lo:$col_hi")
    println("  Vertical bonds: $(length(D_v)), range [$(minimum(D_v)), $(maximum(D_v))]")
    println("  Horizontal bonds: $(length(D_h)), range [$(minimum(D_h)), $(maximum(D_h))]")
    println("  Color scale: ±$cmax")

    plot_title = isempty(title) ?
        "DMRG Bond Energy ⟨Sᵢ·Sⱼ⟩ (Lx=$Lx, Ly=$Ly)" : title

    fig = Figure(size=(max(800, (col_hi - col_lo + 2) * 60), max(400, Ly * 100 + 100)))
    ax = Axis(fig[1, 1], xlabel="Column", ylabel="Row",
              title=plot_title, aspect=DataAspect(), yticks=1:Ly)

    # Draw vertical bonds
    for (idx, (_, _, col, row)) in enumerate(bonds_v)
        row2 = row % Ly + 1
        val = D_v[idx]
        c_norm = val / cmax
        lw = 1.0 + 4.0 * abs(val) / cmax
        if row2 < row
            linesegments!(ax, [Float64(col), Float64(col)],
                          [Float64(row), Float64(row) + 0.5],
                          color=[c_norm, c_norm], colorrange=(-1, 1),
                          colormap=:RdBu, linewidth=lw)
            linesegments!(ax, [Float64(col), Float64(col)],
                          [Float64(row2) - 0.5, Float64(row2)],
                          color=[c_norm, c_norm], colorrange=(-1, 1),
                          colormap=:RdBu, linewidth=lw)
        else
            linesegments!(ax, [Float64(col), Float64(col)],
                          [Float64(row), Float64(row2)],
                          color=[c_norm, c_norm], colorrange=(-1, 1),
                          colormap=:RdBu, linewidth=lw)
        end
    end

    # Draw horizontal bonds
    for (idx, (_, _, col, row)) in enumerate(bonds_h)
        val = D_h[idx]
        c_norm = val / cmax
        lw = 1.0 + 4.0 * abs(val) / cmax
        linesegments!(ax, [Float64(col), Float64(col + 1)],
                      [Float64(row), Float64(row)],
                      color=[c_norm, c_norm], colorrange=(-1, 1),
                      colormap=:RdBu, linewidth=lw)
    end

    # Draw sites
    for col in col_lo:col_hi, row in 1:Ly
        scatter!(ax, [Float64(col)], [Float64(row)], color=:gray40, markersize=6)
    end

    Colorbar(fig[1, 2], colormap=:RdBu, limits=(-cmax, cmax),
             label="⟨Sᵢ · Sⱼ⟩")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    bond_data = (SdotS_matrix=SdotS, bonds_v=bonds_v, bonds_h=bonds_h,
                 D_v=D_v, D_h=D_h)
    return (fig, bond_data)
end
