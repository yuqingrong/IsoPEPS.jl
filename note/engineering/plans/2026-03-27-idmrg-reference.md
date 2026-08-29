# iDMRG Thermodynamic Limit Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a converged iDMRG reference script that saves ground truth energies to JSON, usable as `pepskit_file` overlay in existing visualization code.

**Architecture:** Rewrite `project/mpskit_j1j2.jl` to scan J2 values at multiple MPS bond dimensions, detect convergence, and save results in the existing PEPSKit JSON format (`g_values`/`energies` keys) so the existing `plot_energy_error_vs_scan` function can load them directly as reference.

**Tech Stack:** MPSKit (VUMPS), MPSKitModels (InfiniteCylinder, S_exchange), TensorKit, JSON3

---

### Task 1: Rewrite `project/mpskit_j1j2.jl` with converged reference scan and JSON output

**Files:**
- Modify: `project/mpskit_j1j2.jl`

This is the main deliverable. The script runs VUMPS at increasing MPS D until the energy converges, then saves results in a JSON format compatible with the existing `pepskit_file` parameter in `src/visualization.jl`.

- [ ] **Step 1: Replace the contents of `project/mpskit_j1j2.jl`**

Write the complete new script:

```julia
using IsoPEPS
using TensorKit, MPSKit, MPSKitModels
using JSON3

# =============================================================================
# Converged iDMRG reference for Heisenberg J1-J2 on InfiniteCylinder
# =============================================================================

"""
    run_idmrg_reference(;
        J1=1.0, row=4, unit_cell_cols=2,
        J2_values=0.0:0.1:1.0,
        D_values=[16, 32, 64, 128, 256],
        convergence_tol=1e-6,
        output_file="idmrg_reference.json"
    )

Run VUMPS at increasing MPS bond dimension D until energy per site converges
(|E(D) - E(D_prev)| < convergence_tol). Saves results to JSON.

The output JSON is compatible with the existing `pepskit_file` format used by
`plot_energy_error_vs_scan` in `src/visualization.jl`.
"""
function run_idmrg_reference(;
    J1::Float64=1.0,
    row::Int=4,
    unit_cell_cols::Int=2,
    J2_values=0.0:0.1:1.0,
    D_values::Vector{Int}=[16, 32, 64, 128, 256],
    convergence_tol::Float64=1e-6,
    output_file::String="idmrg_reference.json"
)
    d = 2  # spin-1/2

    # Storage for all results
    all_energies = Dict{Int, Vector{Float64}}()   # D => [E(J2_1), E(J2_2), ...]
    all_corr_lengths = Dict{Int, Vector{Float64}}()
    all_entropies = Dict{Int, Vector{Float64}}()
    converged_energies = Float64[]
    converged_corr_lengths = Float64[]
    converged_entropies = Float64[]
    converged_D = Int[]

    J2_vec = collect(Float64, J2_values)

    for D in D_values
        all_energies[D] = Float64[]
        all_corr_lengths[D] = Float64[]
        all_entropies[D] = Float64[]
    end

    println("=" ^ 70)
    println("iDMRG Reference: Heisenberg J1-J2 on InfiniteCylinder($row, $(row*unit_cell_cols))")
    println("J1=$J1, D_values=$D_values, convergence_tol=$convergence_tol")
    println("=" ^ 70)

    for (i, J2) in enumerate(J2_vec)
        println("\n--- J2 = $J2 ---")
        prev_energy = NaN
        best_energy = NaN
        best_corr = NaN
        best_entropy = NaN
        best_D = D_values[1]

        for D in D_values
            print("  D=$D: ")
            try
                result = mpskit_ground_state_j1j2(d, D, J1, J2, row;
                    unit_cell_cols=unit_cell_cols)
                E = result.energy
                ξ = result.correlation_length
                S = isempty(result.entropy) ? NaN : first(result.entropy)

                push!(all_energies[D], E)
                push!(all_corr_lengths[D], ξ)
                push!(all_entropies[D], S)

                println("E=$E, ξ=$ξ")

                best_energy = E
                best_corr = ξ
                best_entropy = S
                best_D = D

                # Check convergence
                if !isnan(prev_energy) && abs(E - prev_energy) < convergence_tol
                    println("  ✓ Converged at D=$D (ΔE=$(abs(E - prev_energy)))")
                    # Fill remaining D values with NaN for consistency
                    remaining_Ds = D_values[findfirst(==(D), D_values)+1:end]
                    for D_skip in remaining_Ds
                        push!(all_energies[D_skip], NaN)
                        push!(all_corr_lengths[D_skip], NaN)
                        push!(all_entropies[D_skip], NaN)
                    end
                    break
                end
                prev_energy = E
            catch e
                @warn "VUMPS failed at D=$D, J2=$J2: $e"
                push!(all_energies[D], NaN)
                push!(all_corr_lengths[D], NaN)
                push!(all_entropies[D], NaN)
                # Fill remaining D values
                remaining_Ds = D_values[findfirst(==(D), D_values)+1:end]
                for D_skip in remaining_Ds
                    push!(all_energies[D_skip], NaN)
                    push!(all_corr_lengths[D_skip], NaN)
                    push!(all_entropies[D_skip], NaN)
                end
                break
            end
        end

        push!(converged_energies, best_energy)
        push!(converged_corr_lengths, best_corr)
        push!(converged_entropies, best_entropy)
        push!(converged_D, best_D)
    end

    # Save results in pepskit_file-compatible format
    # The existing visualization code expects "g_values" and "energies" keys.
    # For Heisenberg J1-J2, we use J2 as the scan parameter but store under
    # "g_values" for compatibility with plot_energy_error_vs_scan.
    output = Dict{String, Any}(
        "g_values" => J2_vec,
        "energies" => converged_energies,
        "parameters" => Dict(
            "model" => "heisenberg_j1j2",
            "J1" => J1,
            "Ly" => row,
            "unit_cell_cols" => unit_cell_cols,
            "method" => "iDMRG_VUMPS",
            "D_values" => D_values,
            "convergence_tol" => convergence_tol
        ),
        "D_convergence" => Dict(
            "D_values" => D_values,
            "energies_by_D" => Dict(string(D) => all_energies[D] for D in D_values),
            "converged_at_D" => converged_D
        ),
        "correlation_lengths" => converged_corr_lengths,
        "entropies" => converged_entropies
    )

    dir = dirname(output_file)
    !isempty(dir) && !isdir(dir) && mkpath(dir)

    open(output_file, "w") do io
        JSON3.pretty(io, output)
    end
    println("\n" * "=" ^ 70)
    println("Results saved to: $output_file")
    println("=" ^ 70)

    return output
end

# =============================================================================
# Run the reference calculation
# =============================================================================

output = run_idmrg_reference(;
    J1=1.0,
    row=4,
    unit_cell_cols=2,
    J2_values=collect(0.0:0.1:1.0),
    D_values=[16, 32, 64, 128, 256],
    convergence_tol=1e-6,
    output_file=joinpath(@__DIR__, "results", "idmrg_reference_j1j2_Ly=4.json")
)
```

- [ ] **Step 2: Verify the script runs for a single J2 value**

Run a quick test with just one J2 value and small D:

```bash
julia --project=. -e '
using IsoPEPS, TensorKit, MPSKit, MPSKitModels, JSON3
include("project/mpskit_j1j2.jl")
' 2>&1 | head -50
```

The full scan will take time. To do a quick smoke test, temporarily modify the bottom of the script to use `J2_values=[0.5]` and `D_values=[16, 32]` and verify it produces valid JSON output.

- [ ] **Step 3: Verify the JSON output is readable by existing visualization code**

After running, verify the JSON has the expected structure:

```bash
julia --project=. -e '
using JSON3
data = open("project/results/idmrg_reference_j1j2_Ly=4.json", "r") do io
    JSON3.read(io, Dict)
end
println("Keys: ", keys(data))
println("g_values: ", data["g_values"])
println("energies: ", data["energies"])
println("Converged at D: ", data["D_convergence"]["converged_at_D"])
'
```

Expected: JSON loads successfully with `g_values` and `energies` arrays.

- [ ] **Step 4: Commit**

```bash
git add project/mpskit_j1j2.jl
git commit -m "feat: rewrite mpskit_j1j2.jl with converged iDMRG reference scan

Runs VUMPS at increasing MPS bond dimensions until energy converges.
Saves results in pepskit_file-compatible JSON format for use as
ground truth reference in quantum channel comparison plots."
```

---

### Task 2: Test the reference against existing visualization

**Files:**
- No files modified — this is a verification task

- [ ] **Step 1: Run the existing `plot_energy_error_vs_scan` with the new reference**

After the reference JSON exists, verify it works with the existing visualization:

```bash
julia --project=. -e '
using IsoPEPS, CairoMakie, JSON3

# Load a quantum channel result and overlay the iDMRG reference
# (Adjust path to match an existing quantum channel result file)
include("project/postprocess.jl")

# Check that the pepskit_file parameter works with the new JSON
ref_file = "project/results/idmrg_reference_j1j2_Ly=4.json"
data = open(ref_file, "r") do io
    JSON3.read(io, Dict)
end
println("Reference energies loaded:")
for (j2, e) in zip(data["g_values"], data["energies"])
    println("  J2=$j2 → E/N=$e")
end
'
```

- [ ] **Step 2: If quantum channel results exist, create a comparison plot**

```bash
julia --project=. -e '
using IsoPEPS, CairoMakie
include("project/postprocess.jl")

# Example: overlay iDMRG reference on energy scan
# plot_energy_error_vs_scan("project/results";
#     model="heisenberg_j1j2", J1=1.0,
#     pepskit_file="project/results/idmrg_reference_j1j2_Ly=4.json",
#     save_path="project/results/figures/energy_vs_j2_with_idmrg_ref.pdf"
# )
println("Verification complete — reference JSON is compatible with existing visualization.")
'
```

This confirms the end-to-end pipeline: iDMRG reference → JSON → plot overlay.
