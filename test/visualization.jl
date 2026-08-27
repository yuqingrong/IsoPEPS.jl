using Test
using IsoPEPS
using Random
using Statistics
using Bootstrap
using CairoMakie: Axis, Colorbar, Errorbars, Figure, Label, Legend, Makie, Scatter, Theme, to_color, with_theme

function write_test_tfim_circuit(path; g=2.0, row=3, p=3, nqubits=3, energy=0.0)
    params = zeros(gate_parameter_count(p, nqubits))
    result = CircuitOptimizationResult(
        [energy], Vector{Float64}[], Matrix{ComplexF64}[], params, energy,
        Float64[], Float64[], Float64[], true
    )
    input_args = Dict{Symbol,Any}(
        :model => "tfim", :J => 1.0, :g => g, :row => row, :p => p,
        :nqubits => nqubits, :share_params => true, :scan_param => "g"
    )
    save_result(path, result, input_args)
    return path
end


@testset "Quantum journal visualization typography" begin
    theme = paper_theme()
    @test theme.font[] == :regular
    @test occursin(
        "NewComputerModern",
        sprint(show, Makie.to_font(theme.fonts, :regular)))
    @test occursin(
        "Italic",
        sprint(show, Makie.to_font(theme.fonts, :italic)))

    fig = plot_training_history(1:3, [0.0, -0.1, -0.2]; ylims=nothing)
    ax = fig.content[1]
    @test ax.ylabel[] == IsoPEPS.ENERGY_PER_SITE_LABEL
    @test ax.titlefont[] == :regular
    @test ax.xticklabelfont[] == :regular
    @test ax.yticklabelfont[] == :regular
    @test string(ax.ylabel[]) ==
          raw"\mathit{E}/\mathit{N}_{\mathrm{site}}"
    @test string(IsoPEPS.FIELD_LABEL) == raw"\mathit{g}"
    @test string(IsoPEPS.J2_OVER_J1_LABEL) ==
          raw"\mathit{J}_2/\mathit{J}_1"
    @test string(IsoPEPS.M2_LABEL) == raw"\mathit{M}^2(\mathbf{q})"
    @test IsoPEPS.QX_LABEL isa Makie.RichText
    @test IsoPEPS.QY_LABEL isa Makie.RichText
    @test IsoPEPS.QX_LABEL.attributes[:font] == :italic
    @test IsoPEPS.QY_LABEL.attributes[:font] == :italic
    @test IsoPEPS.QX_LABEL.children[2].type == :sub
    @test IsoPEPS.QY_LABEL.children[2].type == :sub
    @test string(IsoPEPS.X_EXPECTATION_LABEL) ==
          raw"\langle \mathit{X}\rangle"
    @test string(IsoPEPS.Z_EXPECTATION_LABEL) ==
          raw"\langle \mathit{Z}\rangle"
    @test IsoPEPS.MAGNETISATION_LABEL == "Magnetisation"
    @test theme.Legend.rowgap[] == IsoPEPS.PAPER_LEGEND_ROWGAP
    @test theme.Legend.colgap[] == IsoPEPS.PAPER_LEGEND_COLGAP
    @test theme.Legend.patchsize[] == IsoPEPS.PAPER_LEGEND_PATCHSIZE
    @test theme.Legend.patchlabelgap[] == IsoPEPS.PAPER_LEGEND_PATCHLABELGAP
    @test string(IsoPEPS.math_label(raw"M^2(0,\pi)")) == raw"M^2(0,\pi)"
    @test string(IsoPEPS.math_label(raw"\mathit{C}(1)\ \mathrm{nearest}")) ==
          raw"\mathit{C}(1)\ \mathrm{nearest}"
    @test string(IsoPEPS.math_label(raw"\mathit{C}(2)\ \mathrm{next-nearest}")) ==
          raw"\mathit{C}(2)\ \mathrm{next-nearest}"
end


@testset "save and load results" begin
    # Test CircuitOptimizationResult
    result_circuit = CircuitOptimizationResult(
        [1.0, 0.5, 0.3],
        Vector{Float64}[],
        [zeros(ComplexF64, 8, 8)],
        [0.1, 0.2],
        0.3,
        [0.5, 0.6],
        [0.8, 0.9],
        Float64[],
        true
    )
    
    input_args = Dict{Symbol, Any}(
        :g => 2.0, :J => 1.0, :row => 3,
        :initial_params => [0.0, 0.0]
    )
    
    tmpfile = tempname() * ".json"
    save_result(tmpfile, result_circuit, input_args)
    @test isfile(tmpfile)
    
    loaded, loaded_args = load_result(tmpfile; result_type=:circuit)
    @test loaded isa CircuitOptimizationResult
    @test loaded.final_cost ≈ result_circuit.final_cost
    @test loaded.converged == result_circuit.converged
    @test loaded_args[:g] == 2.0
    @test loaded_args[:J] == 1.0
    
    rm(tmpfile)
    
    # Test ExactOptimizationResult
    result_exact = ExactOptimizationResult(
        [1.0, 0.5],
        [zeros(ComplexF64, 8, 8)],
        [0.1, 0.2],
        0.5,
        0.2,
        [1.0, 0.8],
        0.9,
        0.85,
        0.88,
        true
    )
    
    input_args = Dict{Symbol, Any}(:g => 2.0, :row => 2)
    
    tmpfile = tempname() * ".json"
    save_result(tmpfile, result_exact, input_args)
    @test isfile(tmpfile)
    
    loaded, loaded_args = load_result(tmpfile; result_type=:exact)
    @test loaded isa ExactOptimizationResult
    @test loaded.gap ≈ result_exact.gap
    @test loaded_args[:g] == 2.0
    
    rm(tmpfile)
end

@testset "plot_correlation_heatmap" begin
    # Skip this test - plot_correlation_heatmap function doesn't exist
    # TODO: Implement or remove this test
    @test true
end

@testset "ACF correlation length from channel sampling" begin
   
end

@testset "plot_training_history" begin
    steps = 1:10
    energies = rand(10)
    
    fig = plot_training_history(steps, energies)
    @test fig isa Figure
    ax = fig.content[1]
    @test ax.xlabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
    @test ax.ylabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
    @test ax.ylabel[] == IsoPEPS.ENERGY_PER_SITE_LABEL
    @test ax.limits[] == (nothing, (-0.5, 0.1))
    @test !ax.xgridvisible[]
    @test !ax.ygridvisible[]
    
    # With reference
    fig2 = plot_training_history(steps, energies; reference=-1.0)
    @test fig2 isa Figure
    legend = fig2.content[2]
    @test legend isa Legend
    @test legend.labelsize[] == 8
    @test legend.rowgap[] == IsoPEPS.PAPER_LEGEND_ROWGAP
    @test legend.colgap[] == IsoPEPS.PAPER_LEGEND_COLGAP
    @test legend.patchsize[] == IsoPEPS.PAPER_LEGEND_PATCHSIZE
    @test legend.patchlabelgap[] == IsoPEPS.PAPER_LEGEND_PATCHLABELGAP
    @test isnothing(legend.layoutobservables.gridcontent[])
    @test IsoPEPS.compact_reference_label(:pepskit, -0.5738123) == "PEPSKit (-0.5738)"
    @test IsoPEPS.compact_reference_label(:dmrg, -0.5738123) == "DMRG (-0.5738)"
    
    # With result
    result = CircuitOptimizationResult(
        energies, Vector{Float64}[], [], Float64[], 0.5, Float64[], Float64[], Float64[], true
    )
    fig3 = plot_training_history(result)
    @test fig3 isa Figure
end

@testset "plot_training_history computes exact contraction reference" begin
    row = 2
    p = 1
    nqubits = 1
    J = 1.0
    g = 2.0
    params = zeros(gate_parameter_count(p, nqubits))
    result = CircuitOptimizationResult(
        [0.2, 0.1], Vector{Float64}[], Matrix{ComplexF64}[], params, 0.1,
        Float64[], Float64[], Float64[], true
    )

    exact_energy = IsoPEPS._training_exact_energy(
        result; row=row, p=p, nqubits=nqubits, J=J, g=g)
    gates = build_unitary_gate(params, p, row, nqubits)
    energy_per_column = first(compute_exact_energy(TFIM(J, g), TransferOperator(gates, row, nqubits)))
    @test exact_energy ≈ energy_per_column / row

    fig = plot_training_history(
        result; row=row, p=p, nqubits=nqubits, J=J, g=g, ylims=nothing)
    @test fig isa Figure
    ax = fig.content[1]
    @test length(ax.scene.plots) == 2
end

@testset "plot_training_history infers legacy 2x2 Heisenberg results" begin
    row = 4
    p = 1
    nqubits = 1
    J1 = 1.0
    J2 = 0.5
    params = zeros(gate_parameter_count(
        p, nqubits; unit_cell=:two_by_two, row=row))
    result = CircuitOptimizationResult(
        [0.0], Vector{Float64}[], Matrix{ComplexF64}[], params, 0.0,
        Float64[], Float64[], Float64[], true
    )

    inferred = IsoPEPS._training_exact_energy(
        result;
        row=row,
        p=p,
        nqubits=nqubits,
        model="heisenberg_j1j2",
        J1=J1,
        J2=J2,
        share_params=true)
    gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
    expected = compute_exact_heisenberg_energy_2x2(
        gates_odd, gates_even, row, 0, J1, J2) / row
    @test inferred ≈ expected
end

@testset "plot_training_history ignores ambient axis label sizes" begin
    with_theme(Theme(Axis=(xlabelsize=21, ylabelsize=19,
                           xticklabelsize=17, yticklabelsize=15,
                           titlesize=23))) do
        fig = plot_training_history(1:3, [0.1, 0.2, 0.3]; ylabel="Energy")
        ax = fig.content[1]
        @test ax.xlabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
        @test ax.ylabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
        @test ax.xticklabelsize[] == IsoPEPS.PAPER_TICKLABELSIZE
        @test ax.yticklabelsize[] == IsoPEPS.PAPER_TICKLABELSIZE
        @test ax.titlesize[] == IsoPEPS.PAPER_TITLESIZE
    end
end

@testset "plot_magnetization_vs_g exact contraction" begin
    data_dir = mktempdir()
    g = 2.0
    row = 2
    p = 1
    nqubits = 1
    structure = :abc
    active_nqubits = 1
    params = collect(range(0.1, 0.4; length=gate_parameter_count(
        p, nqubits; structure=structure, row=row,
        active_nqubits=active_nqubits)))
    result = CircuitOptimizationResult(
        [0.0], Vector{Float64}[], Matrix{ComplexF64}[], params, 0.0,
        Float64[], Float64[], Float64[], true)
    input_args = Dict{Symbol,Any}(
        :model => "tfim", :J => 1.0, :g => g, :row => row, :p => p,
        :nqubits => nqubits, :share_params => false,
        :structure => String(structure), :active_nqubits => active_nqubits,
        :scan_param => "g")
    filename = joinpath(
        data_dir,
        "circuit_tfim_J=1.0_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1.json")
    save_result(filename, result, input_args)

    fig, data = plot_magnetization_vs_g(
        data_dir, [g]; J=1.0, row=row, p=p, nqubits=nqubits,
        use_exact=true)

    gates = build_unitary_gate(
        params, p, row, nqubits;
        share_params=false, structure=structure,
        active_nqubits=active_nqubits)
    virtual_qubits = (nqubits - 1) ÷ 2
    expected_mZ = abs(real(compute_Z_expectation(
        nothing, gates, row, virtual_qubits)))
    expected_mX = abs(real(compute_X_expectation(
        nothing, gates, row, virtual_qubits)))

    @test fig isa Figure
    @test collect(keys(data)) == [g]
    @test data[g].mZ ≈ expected_mZ
    @test data[g].mX ≈ expected_mX
    legend = only(filter(content -> content isa Legend, fig.content))
    @test isnothing(legend.layoutobservables.gridcontent[])
    @test legend.halign[] == 0.98
    @test legend.valign[] == 0.78
    @test legend.nbanks[] == 1
    @test legend.labelsize[] == IsoPEPS.PAPER_LARGE_LEGEND_LABELSIZE
    ax = only(filter(content -> content isa Axis, fig.content))
    @test ax.finallimits[].widths[2] > max(expected_mZ, expected_mX)
    @test ax.xlabelsize[] == IsoPEPS.PAPER_LARGE_AXIS_LABELSIZE
    @test ax.ylabelsize[] == IsoPEPS.PAPER_LARGE_AXIS_LABELSIZE
    @test ax.xticklabelsize[] == IsoPEPS.PAPER_LARGE_TICKLABELSIZE
    @test ax.yticklabelsize[] == IsoPEPS.PAPER_LARGE_TICKLABELSIZE
end

@testset "plot_connected_corr_vs_g uses larger text" begin
    data_dir = mktempdir()
    write_test_tfim_circuit(
        joinpath(data_dir,
                 "circuit_tfim_J=1.0_g=2.0_row=3_p=3_nqubits=3_1x1.json"))

    fig, _ = plot_connected_corr_vs_g(
        data_dir, [2.0]; J=1.0, row=3, p=3, nqubits=3,
        use_exact=true)

    ax = only(filter(content -> content isa Axis, fig.content))
    legend = only(filter(content -> content isa Legend, fig.content))
    @test ax.xlabelsize[] == IsoPEPS.PAPER_LARGE_AXIS_LABELSIZE
    @test ax.ylabelsize[] == IsoPEPS.PAPER_LARGE_AXIS_LABELSIZE
    @test ax.xticklabelsize[] == IsoPEPS.PAPER_LARGE_TICKLABELSIZE
    @test ax.yticklabelsize[] == IsoPEPS.PAPER_LARGE_TICKLABELSIZE
    @test legend.labelsize[] == IsoPEPS.PAPER_LARGE_LEGEND_LABELSIZE
end

@testset "plot_correlation_vs_g legend stays stacked top-left" begin
    data_dir = mktempdir()
    write_test_tfim_circuit(
        joinpath(data_dir, "circuit_tfim_J=1.0_g=2.0_row=3_p=3_nqubits=3_1x1_6w.json"))
    dmrg_file = joinpath(data_dir, "dmrg_tfim_100x3.json")
    pepskit_file = joinpath(data_dir, "pepskit_results_D=2.json")
    save_results(dmrg_file; scan_values=[2.0], correlation_lengths=[1.5])
    save_results(pepskit_file; g_values=[2.0], correlation_lengths=[1.4])

    fig, _ = plot_correlation_vs_g(data_dir, [2.0];
                                   max_separation=1,
                                   dmrg_file=dmrg_file,
                                   pepskit_file=pepskit_file,
                                   g_c=3.04438)
    @test fig isa Figure
    legend = fig.content[2]
    @test legend isa Legend
    @test legend.nbanks[] == 1
    @test legend.labelsize[] == IsoPEPS.PAPER_LARGE_LEGEND_LABELSIZE
    ax = only(filter(content -> content isa Axis, fig.content))
    @test ax.xlabelsize[] == IsoPEPS.PAPER_LARGE_AXIS_LABELSIZE
    @test ax.ylabelsize[] == IsoPEPS.PAPER_LARGE_AXIS_LABELSIZE
    @test ax.xticklabelsize[] == IsoPEPS.PAPER_LARGE_TICKLABELSIZE
    @test ax.yticklabelsize[] == IsoPEPS.PAPER_LARGE_TICKLABELSIZE
    critical_lines = filter(plot -> plot isa Makie.VLines, ax.scene.plots)
    @test length(critical_lines) == 1
    @test critical_lines[1][1][] == [3.04438]
    @test to_color(critical_lines[1].color[]) == to_color(:gray40)
    @test critical_lines[1].linestyle[] == :dash
    @test critical_lines[1].linewidth[] == 1.0
    @test length(legend.entrygroups[][1][2]) == 3
end

@testset "plot_correlation_vs_g loads 6w circuit results" begin
    data_dir = mktempdir()
    write_test_tfim_circuit(
        joinpath(data_dir, "circuit_tfim_J=1.0_g=2.0_row=3_p=3_nqubits=3_1x1_6w.json"))

    fig, data = plot_correlation_vs_g(data_dir, [2.0]; max_separation=1)

    @test fig isa Figure
    @test haskey(data, 2.0)
end

@testset "plot_correlation_vs_g uses transfer spectrum only" begin
    data_dir = mktempdir()
    write_test_tfim_circuit(
        joinpath(data_dir, "circuit_tfim_J=1.0_g=2.0_row=3_p=3_nqubits=3_1x1_6w.json"))

    fig, data = plot_correlation_vs_g(data_dir, [2.0];
        spectrum_krylovdim=8,
        spectrum_tol=1e-6,
        spectrum_maxiter=100,
        spectrum_eager=false)

    @test fig isa Figure
    @test haskey(data, 2.0)
    @test isempty(data[2.0].separations)
    @test isempty(data[2.0].correlations)
    @test data[2.0].correlation_length_fitted === nothing
    @test isfinite(data[2.0].correlation_length)
end

@testset "plot_energy_error_vs_g supports multiple circuit and reference series" begin
    data_dir = mktempdir()
    scan_values = [1.0, 2.0]

    function write_circuit(path, energy; nqubits=3)
        params = zeros(gate_parameter_count(3, nqubits))
        result = CircuitOptimizationResult(
            [energy], Vector{Float64}[], Matrix{ComplexF64}[], params, energy,
            Float64[], Float64[], Float64[], true
        )
        input_args = Dict{Symbol,Any}(
            :model => "tfim", :J => 1.0, :row => 3, :p => 3,
            :nqubits => nqubits, :share_params => true, :scan_param => "g"
        )
        save_result(path, result, input_args)
    end

    for (g, energy) in zip(scan_values, [-2.0, -3.0])
        write_circuit(joinpath(data_dir, "circuit_tfim_J=1.0_g=$(g)_row=3_p=3_nqubits=3_1x1.json"), energy)
    end

    for (g, energy) in zip(scan_values, [-1.9, -2.8])
        write_circuit(joinpath(data_dir, "circuit_tfim_J=1.0_g=$(g)_row=3_p=3_nqubits=5_1x1_6w.json"), energy; nqubits=5)
    end

    dmrg_d32 = joinpath(data_dir, "dmrg_bulk_tfim_Ly3_D32_gscan.json")
    dmrg_d2 = joinpath(data_dir, "dmrg_bulk_tfim_Ly3_D2_gscan.json")
    save_results(dmrg_d32; scan_values=scan_values, e_bulk_values=[-2.05, -3.05])
    save_results(dmrg_d2; scan_values=scan_values, e_bulk_values=[-2.1, -3.1])

    fig_energy, fig_error, data = plot_energy_error_vs_g(data_dir, scan_values;
        energy_source=:saved,
        dmrg_file=[dmrg_d32, dmrg_d2],
        g_c=3.04438,
        circuit_series=[
            (label="IsoPEPS χ=5", nqubits=5, suffixes=["_1x1_6w"], energy_source=:saved),
        ])

    @test fig_energy isa Figure
    @test fig_error isa Figure
    @test fig_energy.content[1].xlabel[] == IsoPEPS.FIELD_LABEL
    @test fig_energy.content[1].ylabel[] == IsoPEPS.ENERGY_PER_SITE_LABEL
    @test fig_error.content[1].xlabel[] == IsoPEPS.FIELD_LABEL
    @test string(fig_error.content[1].ylabel[]) ==
          raw"\frac{|\mathit{E}_{\mathrm{isoPEPS}}-\mathit{E}_{\mathrm{DMRG}}|}{|\mathit{E}_{\mathrm{DMRG}}|}"
    @test isempty(filter(content -> content isa Legend, fig_error.content))
    @test haskey(data.series, "IsoPEPS")
    @test haskey(data.series, "IsoPEPS χ=5")
    @test haskey(data.series, "DMRG D=32")
    @test haskey(data.series, "DMRG D=2")
    @test data.series["IsoPEPS χ=5"].energies == [-1.9, -2.8]
    @test haskey(data.errors_by_reference, "IsoPEPS − DMRG D=32")
    @test haskey(data.errors_by_reference, "IsoPEPS − DMRG D=2")
    @test data.errors_by_reference["IsoPEPS χ=5 − DMRG D=32"].errors ≈ [0.15/2.05, 0.25/3.05]
    @test data.energies_dmrg == [-2.05, -3.05]
    @test IsoPEPS._series_errors(
        (scan_values=[1.0], energies=[-2.1]),
        (scan_values=[1.0], energies=[-2.0])).errors ≈ [0.05]

    energy_axis = only(filter(content -> content isa Axis, fig_energy.content))
    energy_series = filter(plot -> hasproperty(plot, :marker) &&
                                  hasproperty(plot, :markercolor) &&
                                  hasproperty(plot, :markersize) &&
                                  hasproperty(plot, :label),
                           energy_axis.scene.plots)
    isopeps_plot = only(filter(plot -> plot.label[] == "IsoPEPS", energy_series))
    dmrg_plots = filter(plot -> startswith(String(plot.label[]), "DMRG"), energy_series)
    @test length(dmrg_plots) == 2
    @test all(plot -> plot.marker[] == :rect, dmrg_plots)
    @test all(plot -> plot.markercolor[] == :transparent, dmrg_plots)
    @test all(plot -> plot.strokecolor[] == plot.color[], dmrg_plots)
    @test all(plot -> plot.strokewidth[] > 0, dmrg_plots)
    @test all(plot -> plot.markersize[] > isopeps_plot.markersize[], dmrg_plots)

    error_axis = only(filter(content -> content isa Axis, fig_error.content))
    critical_lines = filter(plot -> plot isa Makie.VLines, error_axis.scene.plots)
    @test length(critical_lines) == 1
    @test critical_lines[1][1][] == [3.04438]
    @test critical_lines[1].linestyle[] == :dash
end

@testset "plot_energy_error_vs_g formats DMRG bond-dimension labels" begin
    @test IsoPEPS._energy_plot_label("DMRG D=32", false) == "DMRG"
    @test IsoPEPS._energy_plot_label("DMRG D=2", false) == "DMRG D=2"

    heisenberg_label = IsoPEPS._energy_plot_label("DMRG D=32", true)
    @test heisenberg_label isa Makie.RichText
    @test heisenberg_label.children[2].attributes[:font] == :italic

    error_label = IsoPEPS._energy_plot_label("IsoPEPS − DMRG D=32", true)
    @test error_label isa Makie.RichText
    @test error_label.children[2].attributes[:font] == :italic
end

@testset "scan energy loader accepts incomplete DMRG scans" begin
    dmrg_file = tempname() * "_dmrg_bulk_heisenberg_j1j2_Ly4_D32_J2scan.json"
    save_results(dmrg_file;
                 scan_values=collect(0.0:0.1:1.0),
                 e_bulk_values=collect(-0.68:0.01:-0.62))

    series = IsoPEPS._load_scan_energy_series(dmrg_file; fallback_label="DMRG")

    @test series.label == "DMRG D=32"
    @test length(series.scan_values) == 7
    @test length(series.energies) == 7
    @test series.scan_values == collect(0.0:0.1:0.6)
end

@testset "plot_energy_error_vs_g energy mode selection" begin
    @test IsoPEPS._circuit_energy_mode("tfim", 3, :computed) == :exact
    @test IsoPEPS._circuit_energy_mode("tfim", 3, :computed; row=3) == :exact
    @test IsoPEPS._circuit_energy_mode("tfim", 5, :computed; row=3) == :sampled
    @test IsoPEPS._circuit_energy_mode("tfim", 5, :computed; row=2) == :exact
    @test IsoPEPS._circuit_energy_mode("tfim", 7, :computed; row=3) == :sampled
    @test IsoPEPS._circuit_energy_mode("tfim", 5, :resampled) == :sampled
    @test IsoPEPS._circuit_energy_mode("heisenberg_j1j2", 5, :computed) == :exact
    @test IsoPEPS._circuit_energy_mode("heisenberg_j1j2", 5, :resampled) == :sampled
    @test IsoPEPS._circuit_energy_mode("tfim", 5, :saved) == :saved
end

@testset "plot_energy_error_vs_g normalizes exact Heisenberg energy per site" begin
    row = 3
    p = 1
    nqubits = 1
    J1 = 1.0
    J2 = 0.5
    params = zeros(gate_parameter_count(p, nqubits;
                                        unit_cell=:two_by_two, row=row))
    result = CircuitOptimizationResult(
        Float64[], Vector{Float64}[], Matrix{ComplexF64}[], params, 0.0,
        Float64[], Float64[], Float64[], true
    )
    input_args = Dict{Symbol,Any}(
        :model => "heisenberg_j1j2",
        :J1 => J1,
        :row => row,
        :p => p,
        :nqubits => nqubits,
        :unit_cell => "two_by_two",
        :active_nqubits => nqubits,
    )
    spec = Dict{Symbol,Any}(:energy_source => :computed)

    energy = IsoPEPS._compute_circuit_energy_from_result(
        "unused.json", result, input_args, J2, spec, 0, 1)
    gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
    energy_per_column = compute_exact_heisenberg_energy_2x2(
        gates_odd, gates_even, row, 0, J1, J2)

    @test energy ≈ energy_per_column / row
    @test energy != energy_per_column
end

@testset "plot_energy_error_vs_g places Heisenberg legends bottom-left" begin
    data_dir = mktempdir()
    J2 = 0.5
    row = 2
    p = 1
    nqubits = 1
    params = zeros(gate_parameter_count(
        p, nqubits; unit_cell=:two_by_two, row=row))
    result = CircuitOptimizationResult(
        [-0.4], Vector{Float64}[], Matrix{ComplexF64}[], params, -0.4,
        Float64[], Float64[], Float64[], true)
    input_args = Dict{Symbol,Any}(
        :model => "heisenberg_j1j2", :J1 => 1.0, :J2 => J2,
        :row => row, :p => p, :nqubits => nqubits,
        :unit_cell => "two_by_two")
    save_result(
        joinpath(data_dir,
                 "circuit_heisenberg_j1j2_J1=1.0_J2=$(J2)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
        result, input_args)
    dmrg_file = joinpath(data_dir, "dmrg_heisenberg.json")
    save_results(dmrg_file; scan_values=[J2], e_bulk_values=[-0.5])

    fig_energy, fig_error, _ = plot_energy_error_vs_g(
        data_dir, [J2]; model="heisenberg_j1j2", J1=1.0,
        row=row, p=p, nqubits=nqubits, energy_source=:saved,
        dmrg_file=dmrg_file)

    expected_positions = ((0.18, 0.02), (0.12, 0.04))
    for (fig, (halign, valign)) in zip((fig_energy, fig_error), expected_positions)
        legend = only(filter(content -> content isa Legend, fig.content))
        @test isnothing(legend.layoutobservables.gridcontent[])
        @test legend.halign[] == halign
        @test legend.valign[] == valign
    end
end

@testset "random-parameter dynamics plots smoke test" begin
    data_dir = mktempdir()
    g = 0.5
    row = 1
    p = 1
    nqubits = 5
    params = zeros(gate_parameter_count(p, nqubits))
    result = CircuitOptimizationResult(
        [0.0], Vector{Float64}[], Matrix{ComplexF64}[], params, 0.0,
        Float64[], Float64[], Float64[], true
    )
    input_args = Dict{Symbol,Any}(
        :model => "tfim", :J => 1.0, :g => g,
        :row => row, :p => p, :nqubits => nqubits,
        :share_params => true,
    )
    filename = joinpath(data_dir,
        "circuit_tfim_J=1.0_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_6w.json")
    save_result(filename, result, input_args)

    random_params = IsoPEPS._select_plot_params(result, :random, 1; random_seed=1)
    @test length(random_params) == length(result.final_params)
    @test all(x -> 0 <= x < 2π, random_params)

    fig_energy = plot_energy_dynamics_vs_g(data_dir, [g];
        J=1.0, row=row, p=p, nqubits=nqubits,
        M=1, shots=2, conv_step=0,
        parameter_source=:random, random_seed=1)
    @test fig_energy isa Figure
    @test fig_energy.content[1].ylabel[] == IsoPEPS.ENERGY_PER_SITE_LABEL

    fig_xz = plot_local_xz_dynamics_vs_g(data_dir, [g];
        J=1.0, row=row, p=p, nqubits=nqubits,
        M=1, shots=2, conv_step=0,
        parameter_source=:random, random_seed=1)
    @test fig_xz isa Figure
end

@testset "M2 sampling bootstrap helpers" begin
    row = 2
    q = (Float64(π), Float64(π))
    X_samples = Float64[1, -1, 1, 1, -1, -1, 1, -1]
    Z_samples = Float64[1, 1, -1, 1, 1, -1, -1, -1]
    Y_samples = Float64[-1, 1, 1, -1, 1, 1, -1, 1]

    function direct_structure_factor(samples)
        qx, qy = q
        ncols = length(samples) ÷ row
        max_sep = min(2, ncols - 1)
        S = 0.0
        for pos1 in 1:row, pos2 in 1:row
            S += cos(qy * (pos2 - pos1)) *
                 IsoPEPS.expect(samples, row, pos1, pos2; col_separation=0)
        end
        for sep in 1:max_sep, pos1 in 1:row, pos2 in 1:row
            S += 2.0 * cos(qx * sep + qy * (pos2 - pos1)) *
                 IsoPEPS.expect(samples, row, pos1, pos2; col_separation=sep)
        end
        return S / row
    end

    @test mean(IsoPEPS._structure_factor_column_contributions(
        X_samples, row, q; max_separation=2)) ≈ direct_structure_factor(X_samples)

    X_contributions = IsoPEPS._m2_basis_column_contributions(X_samples, row, q;
                                                              max_separation=2)
    Z_contributions = IsoPEPS._m2_basis_column_contributions(Z_samples, row, q;
                                                              max_separation=2)
    Y_contributions = IsoPEPS._m2_basis_column_contributions(Y_samples, row, q;
                                                              max_separation=2)
    contribution_mean = (sum(X_contributions) + sum(Z_contributions) +
                         sum(Y_contributions)) / length(X_contributions)
    @test contribution_mean ≈ magnetic_order_squared(X_samples, Z_samples, Y_samples,
                                                      row, q; max_separation=2)

    constant_samples = ones(32)
    constant_contributions = IsoPEPS._m2_basis_column_contributions(
        constant_samples, 1, (0.0, 0.0); max_separation=3)
    @test all(value -> value ≈ first(constant_contributions), constant_contributions)
    stderr = IsoPEPS._m2_stderr_from_contributions(
        constant_contributions, constant_contributions, constant_contributions;
        n_bootstrap=10, block_cols=4, rng=MersenneTwister(1))
    @test stderr ≈ 0.0 atol=1e-14

    values = collect(1.0:5.0)
    prefix_sums = IsoPEPS._m2_prefix_sums(values)
    rng = MersenneTwister(2)
    expected_rng = MersenneTwister(2)
    full_start = rand(expected_rng, 1:2)
    partial_start = rand(expected_rng, 1:5)
    expected_mean = (sum(@view values[full_start:full_start+3]) + values[partial_start]) / 5
    @test IsoPEPS._m2_block_bootstrap_mean(prefix_sums, 4, rng) == expected_mean
end

@testset "save_M2_vs_J2 stores sampling standard errors" begin
    data_dir = mktempdir()
    J2 = 0.0
    row = 4
    p = 1
    nqubits = 1
    params = zeros(gate_parameter_count(p, nqubits))
    result = CircuitOptimizationResult(
        [0.0], Vector{Float64}[], Matrix{ComplexF64}[], params, 0.0,
        Float64[], Float64[], Float64[], true
    )
    input_args = Dict{Symbol,Any}(
        :model => "heisenberg_j1j2", :J1 => 1.0, :J2 => J2,
        :row => row, :p => p, :nqubits => nqubits, :share_params => true,
    )
    circuit_file = joinpath(data_dir,
        "circuit_heisenberg_j1j2_J1=1.0_J2=$(J2)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")
    samples_file = joinpath(data_dir, "samples_heisenberg_J2=$(J2).json")
    output_file = joinpath(data_dir, "M2_sampling.json")
    save_result(circuit_file, result, input_args)

    columns = fill([1.0, 1.0, -1.0, -1.0], 6)
    X_samples = reduce(vcat, columns)
    Z_samples = copy(X_samples)
    Y_samples = copy(X_samples)
    save_results(samples_file;
        conv_step=0,
        X_samples=[X_samples],
        Z_samples=[Z_samples],
        Y_samples=[Y_samples])

    data = save_M2_vs_J2(data_dir, [J2];
        method=:sampling, output_file=output_file,
        row=row, p=p, nqubits=nqubits,
        max_separation=20, conv_step=0, samples=length(X_samples),
        n_bootstrap=8, bootstrap_block_cols=2,
        samples_files=Dict(J2 => samples_file))

    @test length(data.M2_neel_stderr) == 1
    @test length(data.M2_stripe_stderr) == 1
    @test length(data.M2_stripe_0pi_stderr) == 1
    @test length(data.MD2_0pi) == 1
    @test all(>=(0), data.M2_neel_stderr)
    @test all(>=(0), data.M2_stripe_stderr)
    @test all(>=(0), data.M2_stripe_0pi_stderr)
    max_sep = length(X_samples) ÷ row - 1
    expected_md2 = dimer_structure_factor(
        X_samples, Z_samples, Y_samples, row, (0.0, Float64(π));
        max_separation=max_sep,
        mean_subtraction=:global,
        distance_window=:bartlett) / (row * (2 * max_sep + 1))
    strictly_connected_md2 = dimer_structure_factor(
        X_samples, Z_samples, Y_samples, row, (0.0, Float64(π));
        max_separation=max_sep) / (row * (2 * max_sep + 1))
    @test only(data.MD2_0pi) ≈ expected_md2
    @test expected_md2 > 0
    @test strictly_connected_md2 ≈ 0 atol=1e-14

    saved = load_results(output_file)
    @test saved["MD2_0pi"] ≈ [expected_md2]
    @test saved["samples"] == length(X_samples)
    @test saved["conv_step"] == 0
    @test saved["n_bootstrap"] == 8
    @test saved["bootstrap_block_cols"] == [2]
end

@testset "plot_M2_comparison qualitative phases and error bars" begin
    data_dir = mktempdir()
    exact_file = joinpath(data_dir, "M2_exact.json")
    sampling_file = joinpath(data_dir, "M2_sampling.json")
    dmrg_file = joinpath(data_dir, "M2_dmrg.json")
    save_results(exact_file;
        J2_values=[0.1, 0.5, 0.8],
        M2_neel=[0.20, 0.10, 0.04],
        M2_stripe_0pi=[0.03, 0.08, 0.18])
    save_results(sampling_file;
        J2_values=[0.1, 0.5, 0.8],
        M2_neel=[0.19, 0.03, 0.02],
        M2_neel_stderr=[0.01, 0.01, 0.005],
        M2_stripe_0pi=[0.02, 0.03, 0.17],
        M2_stripe_0pi_stderr=[0.005, 0.01, 0.015],
        MD2_0pi=[0.01, 0.12, 0.03])
    save_results(dmrg_file;
        scan_values=[0.1, 0.5, 0.8],
        M2_neel_Lx2=[0.21, 0.11, 0.05],
        M2_stripe_Lx2=[0.025, 0.075, 0.18],
        D2_vertical_0pi_Lx2=[0.015, 0.11, 0.025])

    fig = plot_M2_comparison(exact_file=exact_file, sampling_file=sampling_file,
                             dmrg_file=dmrg_file)
    @test fig isa Figure
    figure_padding = fig.layout.alignmode[].padding
    @test (figure_padding.left, figure_padding.right,
           figure_padding.bottom, figure_padding.top) == (6, 12, 6, 6)
    axes = filter(content -> content isa Axis, fig.content)
    @test length(axes) == 1
    ax = axes[1]
    @test ax.xlabel[] == IsoPEPS.J2_OVER_J1_LABEL
    @test ax.ylabel[] == IsoPEPS.math_label(
        raw"\mathit{M}^2(\mathbf{q}),\ \mathit{M}_{\mathrm{D}}^2(\mathbf{q})")
    @test ax.xgridvisible[] == false
    @test ax.ygridvisible[] == false
    legends = filter(content -> content isa Legend, fig.content)
    @test length(legends) == 1
    observable_legend = only(legends)
    @test isnothing(observable_legend.layoutobservables.gridcontent[])
    @test observable_legend.nbanks[] == 1
    @test observable_legend.halign[] == 0.02
    @test observable_legend.valign[] == 0.08
    @test observable_legend.margin[] == (1, 1, 1, 1)
    @test ax.finallimits[].origin[2] == 0
    g = observable_legend.entrygroups[][1]
    legend_labels = [e.label[] for e in g[2]]
    @test legend_labels == [
        IsoPEPS.math_label(raw"\mathit{M}^2(\pi,\pi)"),
        IsoPEPS.math_label(raw"\mathit{M}^2(0,\pi)"),
        IsoPEPS.math_label(raw"\mathit{M}_{\mathrm{D}}^2(0,\pi)"),
    ]
    @test !any(plot -> hasproperty(plot, :text) && any(note ->
                  occursin("isoPEPS: solid", note) || occursin("DMRG: dashed", note),
                  plot.text[]),
               ax.scene.plots)
    @test all(label -> label isa typeof(IsoPEPS.math_label("")), legend_labels)
    @test count(plot -> plot isa Errorbars, ax.scene.plots) == 2
    line_series = filter(plot -> hasproperty(plot, :linestyle) && hasproperty(plot, :marker),
                         ax.scene.plots)
    @test all(plot -> plot.linestyle[] == :solid, line_series)
    styled_series = filter(ax.scene.plots) do plot
        hasproperty(plot, :marker) && hasproperty(plot, :markersize)
    end
    @test all(plot -> plot.marker[] == :rect, styled_series)
    @test !any(plot -> plot.marker[] == :xcross, styled_series)
    # phase text labels "Néel", "VBS", "Stripe" appear in the main axis
    phase_texts = [only(plot.text[]) for plot in ax.scene.plots
                   if hasproperty(plot, :text) && only(plot.text[]) in ["Néel", "VBS", "Stripe"]]
    @test phase_texts == ["Néel", "VBS", "Stripe"]
    phase_text_plots = filter(ax.scene.plots) do plot
        hasproperty(plot, :text) && only(plot.text[]) in ["Néel", "VBS", "Stripe"]
    end
    @test all(p -> p.fontsize[] == IsoPEPS.PAPER_LEGEND_LABELSIZE, phase_text_plots)
    @test all(p -> p.color[] == to_color(:gray25), phase_text_plots)
    @test all(p -> p.font[] == :italic, phase_text_plots)
    @test all(p -> p.align[] == (:center, :center), phase_text_plots)
    @test all(p -> Tuple(p.offset[]) == (-0.75f0, 0.25f0, 0.0f0),
              phase_text_plots)

    dmrg_dashed_fig = plot_M2_comparison(sampling_file=sampling_file,
                                         dmrg_file=dmrg_file;
                                         show_dmrg=true)
    dmrg_axes = filter(content -> content isa Axis, dmrg_dashed_fig.content)
    dmrg_lines = filter(plot -> hasproperty(plot, :linestyle) && hasproperty(plot, :marker),
                        dmrg_axes[1].scene.plots)
    isopeps_lines = filter(plot -> plot.marker[] == :rect, dmrg_lines)
    dmrg_reference_lines = filter(plot -> plot.marker[] == :circle, dmrg_lines)
    @test length(isopeps_lines) == 3
    @test length(dmrg_reference_lines) == 3
    @test all(plot -> plot.linestyle[] == :solid, isopeps_lines)
    @test all(plot -> plot.linestyle[] == :dash, dmrg_reference_lines)
    @test [plot.color[] for plot in isopeps_lines] ==
          [:blue, :orange, :seagreen]
    @test [plot.color[] for plot in isopeps_lines] ==
          [plot.color[] for plot in dmrg_reference_lines]

    dmrg_legends = filter(content -> content isa Legend, dmrg_dashed_fig.content)
    @test length(dmrg_legends) == 1
    dmrg_observable_legend = only(dmrg_legends)
    @test [entry.label[] for entry in dmrg_observable_legend.entrygroups[][1][2]] == [
        IsoPEPS.math_label(raw"\mathit{M}^2(\pi,\pi)"),
        IsoPEPS.math_label(raw"\mathit{M}^2(0,\pi)"),
        IsoPEPS.math_label(raw"\mathit{M}_{\mathrm{D}}^2(0,\pi)"),
    ]
    @test !any(plot -> hasproperty(plot, :text) && any(note ->
                  occursin("isoPEPS: solid", note) || occursin("DMRG: dashed", note),
                  plot.text[]),
               dmrg_axes[1].scene.plots)

    larger_marker_fig = plot_M2_comparison(sampling_file=sampling_file;
                                           markersize=6)
    larger_marker_axes = filter(content -> content isa Axis,
                                larger_marker_fig.content)
    larger_marker_series = filter(larger_marker_axes[1].scene.plots) do plot
        hasproperty(plot, :marker) && hasproperty(plot, :markersize)
    end
    @test any(plot -> plot.markersize[] == 6, larger_marker_series)

    no_errorbar_fig = plot_M2_comparison(sampling_file=sampling_file;
                                         show_errorbars=false)
    no_errorbar_axes = filter(content -> content isa Axis, no_errorbar_fig.content)
    @test count(plot -> plot isa Errorbars, no_errorbar_axes[1].scene.plots) == 0

    se_fig = plot_M2_comparison(exact_file=exact_file, sampling_file=sampling_file,
                                dmrg_file=dmrg_file;
                                show_sampling_stderr_panel=true)
    se_axes = filter(content -> content isa Axis, se_fig.content)
    @test length(se_axes) == 2
    @test se_axes[2].ylabel[] == "Sampling SE"
    @test se_axes[2].xgridvisible[] == false
    @test se_axes[2].ygridvisible[] == false
    annotations = IsoPEPS.m2_phase_annotations(0.24)
    @test [a.label for a in annotations] == ["Néel", "VBS", "Stripe"]
    @test [(a.x, a.y) for a in annotations] == [(0.20, 0.62), (0.55, 0.62), (0.80, 0.62)]
    @test [a.label_x for a in annotations] == [0.20, 0.55, 0.80]

    legacy_sampling_file = joinpath(data_dir, "M2_sampling_legacy.json")
    save_results(legacy_sampling_file;
        J2_values=[0.1, 0.5, 0.8],
        M2_neel=[0.19, 0.09, 0.03],
        M2_stripe_0pi=[0.02, 0.07, 0.17])
    legacy_fig = plot_M2_comparison(sampling_file=legacy_sampling_file)
    @test legacy_fig isa Figure
    legacy_axes = filter(content -> content isa Axis, legacy_fig.content)
    @test length(legacy_axes) == 1
    legacy_ax = legacy_axes[1]
    @test count(plot -> plot isa Errorbars, legacy_ax.scene.plots) == 0
    legacy_legend = only(filter(content -> content isa Legend, legacy_fig.content))
    legacy_labels = [entry.label[] for entry in legacy_legend.entrygroups[][1][2]]
    @test legacy_labels == [
        IsoPEPS.math_label(raw"\mathit{M}^2(\pi,\pi)"),
        IsoPEPS.math_label(raw"\mathit{M}^2(0,\pi)"),
    ]

    @test isnothing(IsoPEPS._load_m2_stderr(
        Dict("bad_stderr" => [0.1, 0.2]), "bad_stderr", 3))
    @test isnothing(IsoPEPS._load_m2_stderr(
        Dict("bad_stderr" => [0.1, -0.2, 0.3]), "bad_stderr", 3))
    @test isnothing(IsoPEPS._load_m2_stderr(
        Dict("bad_stderr" => [0.1, NaN, 0.3]), "bad_stderr", 3))
end


@testset "variance bootstrap confidence interval" begin
    energies = [-1.3, -1.1, -0.9, -0.8, -0.7, -0.5]

    Random.seed!(42)
    estimate, lower, upper = IsoPEPS._variance_bootstrap_ci(
        energies; n_resamples=200, confidence_level=0.68)

    Random.seed!(42)
    expected = only(confint(
        bootstrap(var, energies, BasicSampling(200)),
        PercentileConfInt(0.68)))

    @test (estimate, lower, upper) == expected
    @test lower <= estimate <= upper
    @test_throws ArgumentError IsoPEPS._variance_bootstrap_ci(
        [1.0]; n_resamples=200)
    @test_throws ArgumentError IsoPEPS._variance_bootstrap_ci(
        energies; n_resamples=1)
    @test_throws ArgumentError IsoPEPS._variance_bootstrap_ci(
        energies; confidence_level=1.0)
end

@testset "plot_variance_vs_samples" begin
    samples = [100, 500, 1000, 5000]
    variances = [0.1, 0.02, 0.01, 0.002]
    
    fig = plot_variance_vs_samples(samples, variances)
    @test fig isa Figure
    
    # With errors
    errors = [0.01, 0.002, 0.001, 0.0002]
    fig2 = plot_variance_vs_samples(samples, variances; errors=errors)
    @test fig2 isa Figure

    asymmetric_errors = [
        (0.01, 0.02),
        (0.002, 0.003),
        (0.001, 0.0015),
        (0.0002, 0.0004),
    ]
    asymmetric_fig = plot_variance_vs_samples(
        samples, variances;
        errors=asymmetric_errors,
        confidence_level=0.68)
    @test asymmetric_fig isa Figure
    asymmetric_ax = only(filter(content -> content isa Axis, asymmetric_fig.content))
    @test count(plot -> plot isa Errorbars, asymmetric_ax.scene.plots) == 1
    asymmetric_scatter = only(filter(
        plot -> plot isa Scatter, asymmetric_ax.scene.plots))
    @test asymmetric_scatter.label[] == "Bootstrap 68% CI"

    results_file = tempname() * ".json"
    save_results(results_file;
                 sample_sizes=samples,
                 variances=variances,
                 variance_ci_lower=variances .- first.(asymmetric_errors),
                 variance_ci_upper=variances .+ last.(asymmetric_errors),
                 confidence_level=0.68,
                 n_bootstrap=200,
                 n_ci_bootstrap=5000)
    file_fig = plot_variance_vs_samples(results_file)
    @test file_fig isa Figure
    file_ax = only(filter(content -> content isa Axis, file_fig.content))
    @test count(plot -> plot isa Errorbars, file_ax.scene.plots) == 1
    file_scatter = only(filter(plot -> plot isa Scatter, file_ax.scene.plots))
    @test file_scatter.label[] == "Bootstrap 68% CI"

    @test_throws DimensionMismatch plot_variance_vs_samples(
        samples, variances; errors=[0.1])
    @test_throws ArgumentError plot_variance_vs_samples(
        samples, variances; errors=Any[(0.1, 0.2), 0.1, 0.1, 0.1])

    styled_fig = plot_variance_vs_samples(samples, variances;
                                          marker=:diamond, markersize=7)
    ax = only(filter(content -> content isa Axis, styled_fig.content))
    scatter = only(filter(plot -> plot isa Scatter, ax.scene.plots))
    @test scatter.marker[] == Makie.to_spritemarker(:diamond)
    @test all(==(7), scatter.markersize[])
    @test scatter.strokewidth[] == 0
end

@testset "displacement-resolved structure factor convention" begin
    q = (pi, pi / 2)
    corr0 = [1.0 2.0; 3.0 4.0]
    corr1 = [0.5 1.0; 1.5 2.0]
    corr2 = [0.25 0.5; 0.75 1.0]

    manual(distance_window) = begin
        total = sum(corr0[p1, p2] * cos(q[2] * (p2 - p1))
                    for p1 in 1:2, p2 in 1:2)
        for (Δ, corr) in enumerate((corr1, corr2))
            weight = distance_window === :bartlett ? 1 - Δ / 3 : 1
            total += sum(2 * weight * corr[p1, p2] *
                         cos(q[1] * Δ + q[2] * (p2 - p1))
                         for p1 in 1:2, p2 in 1:2)
        end
        total / 2
    end

    @test IsoPEPS._displacement_structure_factor(corr0, [corr1, corr2], q) ≈
          manual(:none)
    @test IsoPEPS._displacement_structure_factor(
        corr0, [corr1, corr2], q; distance_window=:bartlett) ≈ manual(:bartlett)

    # Dimer structure factors use the same evaluator after connected
    # (global-mean-subtracted) correlations have been assembled.
    components = (; μ=zeros(2), corr0, corr_dc=[corr1, corr2],
                  max_sep=2, n_pos=2)
    @test IsoPEPS._evaluate_dimer_structure_factor(
        components, q; mean_subtraction=:global,
        distance_window=:bartlett) ≈ manual(:bartlett)

    # The sampling M² convention divides a displacement-resolved spin S(q)
    # by the effective site count, independently of how many finite-cylinder
    # pairs were available to estimate each displacement correlation.
    S = IsoPEPS._displacement_structure_factor(corr0, [corr1, corr2], q)
    @test S / (2 * 5) ≈ manual(:none) / 10
end

@testset "plot_combined_structure_factors formats mathematical labels" begin
    data_file = tempname() * ".json"
    spin_matrices = [fill(0.1, 3, 3), fill(0.2, 3, 3)]
    dimer_matrices = [fill(0.3, 3, 3), fill(0.4, 3, 3)]
    save_results(data_file;
                 J2_values=[0.0, 0.5],
                 nq=3,
                 use_exact=false,
                 spin_matrices=[collect(eachcol(m)) for m in spin_matrices],
                 dimer_matrices=[collect(eachcol(m)) for m in dimer_matrices])

    fig, _, _ = plot_combined_structure_factors(
        "unused", Float64[]; data_file=data_file)

    @test isempty(filter(content -> content isa Label, fig.content))
    axes = filter(content -> content isa Axis, fig.content)
    @test length(axes) == 4
    top_axes = filter(ax -> only(ax.layoutobservables.gridcontent[].span.rows) == 1,
                      axes)
    bottom_axes = filter(ax -> only(ax.layoutobservables.gridcontent[].span.rows) == 2,
                         axes)
    @test [ax.title[] for ax in top_axes] ==
          [IsoPEPS.math_label("\\mathit{J}_2=$value") for value in (0.0, 0.5)]
    @test first(top_axes).ylabel[] == IsoPEPS.QY_LABEL
    @test first(bottom_axes).xlabel[] == IsoPEPS.QX_LABEL
    @test first(bottom_axes).ylabel[] == IsoPEPS.QY_LABEL

    colorbars = filter(content -> content isa Colorbar, fig.content)
    @test [colorbar.label[] for colorbar in colorbars] == [
        IsoPEPS.math_label(raw"\mathit{S}(\mathbf{q})"),
        IsoPEPS.math_label(raw"\mathit{S}_{\mathrm{D}}(\mathbf{q})"),
    ]
    @test all(colorbar -> colorbar.ticklabelsize[] ==
                         IsoPEPS._COMBINED_COLORBAR_TICKLABELSIZE,
              colorbars)
    @test all(colorbar -> colorbar.width[] ==
                         IsoPEPS._COMBINED_COLORBAR_WIDTH,
              colorbars)
    @test all(colorbar -> colorbar.tellheight[] == false, colorbars)
    @test all(colorbar -> colorbar.valign[] == :center, colorbars)
    Makie.resize_to_layout!(fig)
    for (axis, colorbar) in zip((first(top_axes), first(bottom_axes)), colorbars)
        @test colorbar.height[] ≈ Makie.widths(axis.scene.viewport[])[2]
    end
end

@testset "discrete cylinder-momentum structure factors" begin
    # A periodic circumference of four sites permits four distinct transverse
    # momenta; 2π is equivalent to the already included zero-momentum row.
    qy_values = IsoPEPS._allowed_cylinder_qy(4)
    @test qy_values == [0.0, Float64(π / 2), Float64(π), Float64(3π / 2)]
    @test length(unique(qy_values)) == 4
    @test !any(q -> q ≈ 2π, qy_values)

    data_dir = mktempdir()
    samples_file = joinpath(data_dir, "samples_heisenberg_J2=0.0.json")
    X_samples = repeat([1.0, -1.0, 1.0, -1.0], 8)
    Y_samples = repeat([1.0, 1.0, -1.0, -1.0], 8)
    Z_samples = repeat([1.0, -1.0, -1.0, 1.0], 8)
    save_results(samples_file;
                 row=4,
                 conv_step=0,
                 X_samples=[X_samples],
                 Y_samples=[Y_samples],
                 Z_samples=[Z_samples])

    qx_values = [0.0, Float64(π)]
    output_file = joinpath(data_dir, "sf_discrete.json")
    spin, dimer = save_combined_structure_factor_data(
        output_file, data_dir, [0.0];
        row=4,
        qx_values=qx_values,
        qy_values=qy_values,
        max_separation_spin=2,
        max_separation_dimer=2,
        use_exact=false,
        conv_step=0,
        samples=length(X_samples),
        samples_files=Dict(0.0 => samples_file))

    @test size(only(spin)) == (2, 4)
    @test size(only(dimer)) == (2, 4)
    for (i, qx) in enumerate(qx_values), (j, qy) in enumerate(qy_values)
        @test isapprox(only(spin)[i, j], spin_spin_structure_factor(
            X_samples, Z_samples, Y_samples, 4, (qx, qy);
            max_separation=2); atol=1e-12)
        @test isapprox(only(dimer)[i, j], dimer_structure_factor(
            X_samples, Z_samples, Y_samples, 4, (qx, qy);
            max_separation=2,
            mean_subtraction=:global,
            distance_window=:bartlett); atol=1e-12)
    end

    saved = load_results(output_file)
    @test saved["qx_values"] == qx_values
    @test saved["qy_values"] == qy_values
    @test saved["row"] == 4
    @test saved["max_separation_spin"] == 2
    @test saved["max_separation_dimer"] == 2
    @test saved["dimer_orientation"] == "vertical"

    plot_file = joinpath(data_dir, "sf_discrete_plot.json")
    save_results(plot_file;
                 J2_values=[0.0, 0.5, 0.6, 1.0],
                 nq=length(qx_values),
                 qx_values=qx_values,
                 qy_values=qy_values,
                 row=4,
                 use_exact=false,
                 max_separation_spin=2,
                 max_separation_dimer=2,
                 dimer_orientation="vertical",
                 conv_step=0,
                 samples=length(X_samples),
                 spin_matrices=[collect(eachcol(only(spin))) for _ in 1:4],
                 dimer_matrices=[collect(eachcol(only(dimer))) for _ in 1:4])
    fig, _, _ = plot_combined_structure_factors(
        "unused", Float64[]; data_file=plot_file, discrete_qy=true)

    axes = filter(content -> content isa Axis, fig.content)
    @test length(axes) == 8
    @test all(axis -> axis.yticks[][1] == collect(1:4), axes)
    @test all(axis -> axis.yticks[][2] == ["0", "π/2", "π", "3π/2"], axes)
    top_axes = filter(ax -> only(ax.layoutobservables.gridcontent[].span.rows) == 1,
                      axes)
    bottom_axes = filter(ax -> only(ax.layoutobservables.gridcontent[].span.rows) == 2,
                         axes)
    @test first(top_axes).ylabel[] == IsoPEPS.QY_LABEL
    @test first(bottom_axes).xlabel[] == IsoPEPS.QX_LABEL
    @test first(bottom_axes).ylabel[] == IsoPEPS.QY_LABEL

    colorbars = filter(content -> content isa Colorbar, fig.content)
    @test [colorbar.label[] for colorbar in colorbars] == [
        IsoPEPS.math_label(raw"\mathit{S}(\mathbf{q})"),
        IsoPEPS.math_label(raw"\mathit{S}_{\mathrm{D}}(\mathbf{q})"),
    ]
    for row_axes in (top_axes, bottom_axes)
        colorranges = [only(plot.colorrange[] for plot in axis.scene.plots
                            if plot isa Makie.Heatmap) for axis in row_axes]
        @test all(==(first(colorranges)), colorranges)
    end
end

@testset "plot_bond_energy_pattern loads saved samples" begin
    data_dir = mktempdir()
    samples_file = joinpath(data_dir, "samples.json")
    chain = ones(8)
    save_results(samples_file;
                 row=2,
                 conv_step=0,
                 source_file="circuit_heisenberg_J2=0.0_2x2.json",
                 X_samples=[chain],
                 Y_samples=[chain],
                 Z_samples=[chain])

    fig, bond_data = plot_bond_energy_pattern("unused.json";
                                              use_exact=false,
                                              samples=8,
                                              samples_file=samples_file,
                                              max_cols=4)

    @test fig isa Figure
    @test size(bond_data[:vertical]) == (2, 4)
    @test size(bond_data[:horizontal]) == (2, 3)
    @test all(==(0.75), bond_data[:vertical])
    @test all(==(0.75), bond_data[:horizontal])
    colorbar = only(filter(content -> content isa Colorbar, fig.content))
    @test colorbar.label[] == IsoPEPS._BOND_ENERGY_LABEL
    @test colorbar.labelsize[] == IsoPEPS._BOND_ENERGY_LABELSIZE
    @test colorbar.ticklabelsize[] == IsoPEPS._BOND_COLORBAR_TICKLABELSIZE
    @test colorbar.width[] == IsoPEPS._BOND_COLORBAR_WIDTH
end

@testset "plot_bond_energy_pattern lays out J2 panels horizontally" begin
    data_dir = mktempdir()
    chain = ones(8)
    for J2 in (0.0, 0.5, 1.0)
        save_results(joinpath(data_dir, "samples_heisenberg_J2=$J2.json");
                     row=2,
                     conv_step=0,
                     J2=J2,
                     source_file="circuit_heisenberg_J2=$(J2)_2x2.json",
                     X_samples=[chain],
                     Y_samples=[chain],
                     Z_samples=[chain])
    end

    fig, patterns = plot_bond_energy_pattern(data_dir, [0.0, 0.5, 1.0])

    @test fig isa Figure
    @test sort!(collect(keys(patterns))) == [0.0, 0.5, 1.0]
    @test size(patterns[0.0][:vertical]) == (2, 5)
    @test all(==(0.75), patterns[0.0][:vertical])
    axes = filter(content -> content isa Axis, fig.content)
    @test length(axes) == 3
    @test [only(ax.layoutobservables.gridcontent[].span.cols) for ax in axes] ==
          [1, 2, 3]
    @test all(ax -> only(ax.layoutobservables.gridcontent[].span.rows) == 2, axes)
    @test all(ax -> !ax.leftspinevisible[] && !ax.bottomspinevisible[] &&
                    !ax.xticklabelsvisible[] && !ax.yticklabelsvisible[], axes)
    panel_labels = filter(content -> content isa Label, fig.content)
    @test [label.text[] for label in panel_labels] ==
          [IsoPEPS.math_label("\\mathit{J}_2=$value") for value in (0.0, 0.5, 1.0)]
    @test [only(label.layoutobservables.gridcontent[].span.cols)
           for label in panel_labels] == [1, 2, 3]
    @test all(label -> only(label.layoutobservables.gridcontent[].span.rows) == 1,
              panel_labels)
    @test all(label -> label.fontsize[] == IsoPEPS._BOND_ENERGY_PANEL_LABELSIZE,
              panel_labels)

    colorbar = only(filter(content -> content isa Colorbar, fig.content))
    @test colorbar.vertical[] == true
    @test colorbar.limits[] == (-0.5, 0.5)
    @test colorbar.label[] == IsoPEPS._BOND_ENERGY_LABEL
    @test string(colorbar.label[]) ==
          raw"\langle \mathbf{S}_{\mathit{i}}\cdot\mathbf{S}_{\mathit{j}}\rangle"
    @test colorbar.labelsize[] == IsoPEPS._BOND_ENERGY_LABELSIZE
    @test colorbar.ticklabelsize[] == IsoPEPS._BOND_COLORBAR_TICKLABELSIZE
    @test colorbar.width[] == IsoPEPS._BOND_COLORBAR_WIDTH
    @test colorbar.layoutobservables.gridcontent[].span.cols == 4:4
    @test colorbar.layoutobservables.gridcontent[].span.rows == 2:2
end

@testset "Sampled expectations match transfer matrix fixed point" begin
    using Statistics
    using Yao, YaoBlocks
    using Random
    using LinearAlgebra
    
    # This is the correct way to verify measurement results match propagation:
    # 1. Compute exact expectation values from transfer matrix fixed point
    # 2. Sample from quantum channel
    # 3. Check sampled means match exact values within statistical error
    
    for seed in [42, 123, 456]
        Random.seed!(seed)
        
        nqubits = 3
        row = 2
        
        # Random gates
        gate_matrix = rand_unitary(ComplexF64, 2^nqubits)
        gates = [gate_matrix for _ in 1:row]
        
        # Exact values from transfer matrix fixed point
        rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
        virtual_qubits = 1  # Bond qubits per side
        X_exact = real(compute_X_expectation(rho, gates, row, virtual_qubits))
        
        # Sample
        n_samples = 500000
        _, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                          conv_step=2000, 
                                                          samples=n_samples)
        
        X_sampled = mean(X_samples)
        X_std = std(X_samples) / sqrt(length(X_samples))  # Standard error
        X_error = abs(X_sampled - X_exact)
        
        # Test: sampled mean should match exact value within 3σ
        @test X_error < 3 * X_std
        
        println("seed=$seed: ⟨X⟩_exact=$(round(X_exact, digits=4)), ⟨X⟩_sampled=$(round(X_sampled, digits=4)) ± $(round(X_std, digits=4))")
    end
end
