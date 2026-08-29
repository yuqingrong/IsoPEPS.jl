# Processed-Only Figure Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default paper-figure reproduction command render every numerical figure from canonical processed JSON rather than raw circuit results.

**Architecture:** Add two canonical Heisenberg processed-data products: exact bond-energy panel values and the training-history series plus exact reference energy. The primary `plot` path reads those files together with the existing processed inputs; the separate `compute` path derives the same JSON products from raw circuits into `recomputed-data/`.

**Tech Stack:** Julia 1.10+, IsoPEPS, JSON3, CairoMakie, Test, TOML.

**Spec:** `note/engineering/specs/2026-08-29-processed-only-figure-reproduction-design.md`

## Global Constraints

- `plot` must not perform tensor contractions, sampling, or write JSON data.
- `archive` must retain byte-identical baseline restoration.
- `compute` must write only under `OUTPUT/recomputed-data/`; it must never overwrite staged canonical processed files or render manuscript figures.
- Numerical figure inputs in the staging package belong under `results/<model>/processed/`; `results/reference/` remains separate.
- The two supplied illustrations remain curated baseline assets, not processed numerical data.
- Do not commit, amend, reset, or otherwise alter Git history; the user has not requested a commit.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/visualization/structure_factors.jl` | Compute, serialize, load, and draw multi-panel bond-energy data without raw-circuit access in the draw function. |
| `src/visualization/training_expectations.jl` | Compute, serialize, load, and draw the Heisenberg training-history data without resampling in the draw function. |
| `src/IsoPEPS.jl` | Export the four new compute/plot public APIs. |
| `project/postprocess.jl` | Register raw-to-processed compute targets and make the two manuscript plot targets consume processed files. |
| `repro/figure_manifest.toml` | Declare canonical processed inputs and `data_target`s for the two formerly raw-driven figures. |
| `repro/data_manifest.toml` | Include the two generated source JSON files in the staged Heisenberg processed directory. |
| `repro/stage_data.jl` | Describe the processed-only primary plotting contract in the generated package README. |
| `README.md`, `REPRODUCIBILITY.md` | Document the final three-mode workflow accurately. |
| `test/visualization.jl`, `test/reproducibility.jl` | Test processed-only plotting APIs and manifest/mode contracts. |

### Task 1: Processed bond-energy data API

**Files:**
- Modify: `src/visualization/structure_factors.jl:830-1325`
- Modify: `src/IsoPEPS.jl:213-217`
- Test: `test/visualization.jl` near the existing `plot_bond_energy_pattern` tests

**Interfaces:**
- Consumes: raw `circuit_heisenberg_j1j2_*.json` result files during compute only, and the existing `_draw_bond_energy_panel!` renderer.
- Produces:
  - `compute_bond_energy_pattern_data(results_dir::String, J2_values::AbstractVector{<:Real}; max_cols::Int=5, save_path::Union{String,Nothing}=nothing)::Dict{String,Any}`
  - `load_bond_energy_pattern_data(path::String)::Dict{String,Any}`
  - `plot_bond_energy_pattern_from_processed_data(path::String; figsize=nothing, save_path=nothing)::Figure`
- JSON shape: `schema_version`, `method="exact"`, `j2_values`, `row`, `max_cols`, and `entries`; each entry has `j2`, `vertical`, and `horizontal` numeric matrices tiled to `max_cols`.

- [ ] **Step 1: Write the failing processed-data plot test**

  Add a test that writes a literal two-panel JSON fixture and calls the new draw API:

  ```julia
  @testset "processed bond-energy data draws without circuit results" begin
      path = joinpath(mktempdir(), "bond_energy_exact.json")
      JSON3.write(path, Dict(
          "schema_version" => 1, "method" => "exact",
          "j2_values" => [0.0, 0.5], "row" => 2, "max_cols" => 3,
          "entries" => [
              Dict("j2" => 0.0, "vertical" => fill(-0.25, 2, 3), "horizontal" => fill(-0.10, 2, 2)),
              Dict("j2" => 0.5, "vertical" => fill(-0.40, 2, 3), "horizontal" => fill(0.15, 2, 2)),
          ],
      ))
      fig = plot_bond_energy_pattern_from_processed_data(path)
      @test fig isa Figure
      @test length(filter(content -> content isa Axis, fig.content)) == 2
  end
  ```

- [ ] **Step 2: Run the visualization test to verify it fails**

  Run: `julia --project=. test/runtests.jl visualization`

  Expected: failure because `plot_bond_energy_pattern_from_processed_data` is undefined.

- [ ] **Step 3: Implement the raw-to-processed and processed-to-plot functions**

  Extract the exact branch's tiled `vertical`/`horizontal` matrices from the existing single-file `plot_bond_energy_pattern` implementation into a helper used by `compute_bond_energy_pattern_data`. Serialize `entries` with `JSON3.write`, rejecting a missing file, missing `entries`, nonpositive `row`, or matrices inconsistent with `row`/`max_cols` in the loader.

  Build `plot_bond_energy_pattern_from_processed_data` by reusing `_draw_bond_energy_panel!` and the current multi-panel labels, fixed `(-0.5, 0.5)` colorbar, dimensions, and paper theme:

  ```julia
  data = load_bond_energy_pattern_data(path)
  entries = data["entries"]
  fig = with_theme(paper_theme()) do
      figure = Figure(size=(length(entries) * (Int(data["max_cols"]) * 35 + 10) + 95,
                            Int(data["row"]) * 35 + 61))
      for (panel, entry) in enumerate(entries)
          pattern = Dict(:vertical => Float64.(entry["vertical"]),
                         :horizontal => Float64.(entry["horizontal"]))
          Label(figure[1, panel], math_label("\\mathit{J}_2=$(Float64(entry["j2"]))");
                fontsize=_BOND_ENERGY_PANEL_LABELSIZE, font=PAPER_FONT,
                halign=:center, valign=:bottom, padding=(0, 0, 0, 0))
          axis = Axis(figure[2, panel]; aspect=DataAspect())
          _draw_bond_energy_panel!(axis, pattern; colorrange=(-0.5, 0.5))
          hidedecorations!(axis)
          hidespines!(axis)
          colsize!(figure.layout, panel, Fixed(Int(data["max_cols"]) * 35 + 10))
      end
      Colorbar(figure[2, length(entries) + 1]; colormap=:RdBu,
               limits=(-0.5, 0.5), vertical=true,
               label=_BOND_ENERGY_LABEL, labelsize=_BOND_ENERGY_LABELSIZE,
               ticklabelsize=_BOND_COLORBAR_TICKLABELSIZE,
               width=_BOND_COLORBAR_WIDTH)
      figure
  end
  ```

  Export the three APIs from `src/IsoPEPS.jl`.

- [ ] **Step 4: Run the visualization test to verify it passes**

  Run: `julia --project=. test/runtests.jl visualization`

  Expected: the new fixture test passes; record any unrelated pre-existing visualization expectation failures separately rather than changing them.

### Task 2: Processed Heisenberg training-history API

**Files:**
- Modify: `src/visualization/training_expectations.jl:1-270`
- Modify: `src/IsoPEPS.jl:205-213`
- Test: `test/visualization.jl`

**Interfaces:**
- Consumes: one raw `CircuitOptimizationResult` JSON during compute only; static DMRG/PEPSKit reference JSON paths may be used during plotting.
- Produces:
  - `compute_training_history_data(result_file::String; save_path::Union{String,Nothing}=nothing)::Dict{String,Any}`
  - `load_training_history_data(path::String)::Dict{String,Any}`
  - `plot_training_history_from_processed_data(path::String; pepskit_results_file::Union{String,Nothing}=nothing, dmrg_bulk_file::Union{String,Nothing}=nothing, save_path::Union{String,Nothing}=nothing)::Figure`
- JSON shape: `schema_version`, `model`, `J1`, `J2`, `row`, `p`, `nqubits`, `energy_history`, `steps`, and nullable `exact_energy`.

- [ ] **Step 1: Write the failing training-history fixture test**

  Add a literal fixture that contains no raw parameters or samples:

  ```julia
  @testset "processed training history draws without resampling" begin
      path = joinpath(mktempdir(), "training_history.json")
      JSON3.write(path, Dict(
          "schema_version" => 1, "model" => "heisenberg_j1j2",
          "J1" => 1.0, "J2" => 0.5, "row" => 4, "p" => 3, "nqubits" => 3,
          "steps" => [1, 2, 3], "energy_history" => [-0.10, -0.20, -0.30],
          "exact_energy" => -0.35,
      ))
      fig = plot_training_history_from_processed_data(path)
      @test fig isa Figure
      @test only(filter(content -> content isa Axis, fig.content)).xlabel[] == "Optimization step"
  end
  ```

- [ ] **Step 2: Run the visualization test to verify it fails**

  Run: `julia --project=. test/runtests.jl visualization`

  Expected: failure because `plot_training_history_from_processed_data` is undefined.

- [ ] **Step 3: Implement computation, validation, and processed-data drawing**

  `compute_training_history_data` must call `load_result`, record `1:length(result.energy_history)`, copy only the display metadata listed above, and call the existing `_training_exact_energy` once for a `CircuitOptimizationResult`. It must not generate expectation values or call `resample_circuit`.

  `plot_training_history_from_processed_data` must load and validate equal nonempty `steps`/`energy_history` lengths, then invoke the existing vector overload with the stored exact energy and optional static reference paths:

  ```julia
  data = load_training_history_data(path)
  return plot_training_history(data["steps"], data["energy_history"];
      g=nothing, row=Int(data["row"]), nqubits=Int(data["nqubits"]),
      J2=Float64(data["J2"]), exact_energy=data["exact_energy"],
      pepskit_results_file=pepskit_results_file,
      dmrg_bulk_file=dmrg_bulk_file,
      save_path=save_path)
  ```

  Export the three APIs. Keep the existing `analyze_result` API intact for interactive analysis; it is no longer used by the paper reproduction target.

- [ ] **Step 4: Run the visualization test to verify it passes**

  Run: `julia --project=. test/runtests.jl visualization`

  Expected: both new processed-data fixture tests pass; no test opens a GUI window.

### Task 3: Wire processed-only targets and staging declarations

**Files:**
- Modify: `project/postprocess.jl:140-205,372-401`
- Modify: `repro/figure_manifest.toml:80-115`
- Modify: `repro/data_manifest.toml:79-95`
- Modify: `test/reproducibility.jl:46-85`

**Interfaces:**
- Consumes: APIs produced by Tasks 1 and 2.
- Produces:
  - targets `bond-energy-exact-data` and `heisenberg-training-history-data`
  - canonical paths `results/heisenberg/processed/bond_energy_exact.json` and `results/heisenberg/processed/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json`

- [ ] **Step 1: Write failing manifest/target assertions**

  Extend the existing reproducibility test with literal expected declarations:

  ```julia
  bond_figure = only(filter(fig -> fig["manuscript_image"] == "bond_energy_exact.pdf", figures))
  @test bond_figure["data_target"] == "bond-energy-exact-data"
  @test bond_figure["source_data"] == ["results/heisenberg/processed/bond_energy_exact.json"]

  training_figure = only(filter(fig -> occursin("_training_history.pdf", fig["manuscript_image"]), figures))
  @test training_figure["data_target"] == "heisenberg-training-history-data"
  @test training_figure["source_data"][1] ==
      "results/heisenberg/processed/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json"
  @test all(!haskey(figure, "source_data_sets") for figure in figures)
  @test any(haskey(figure, "compute_source_data_sets") for figure in figures)
  ```

- [ ] **Step 2: Run the focused reproducibility test to verify it fails**

  Run: `julia --project=. test/runtests.jl reproducibility`

  Expected: the new assertions fail because the figure manifest still declares raw circuit inputs and no data target for either figure.

- [ ] **Step 3: Implement target and manifest wiring**

  Add two `postprocess.jl` compute targets that write source-tree processed data to the standard figure-data directory:

  ```julia
  function compute_bond_energy_exact_data_target(; results_root=DEFAULT_RESULTS_ROOT, output_dir=results_root)
      compute_bond_energy_pattern_data(_result_path(results_root, "heisenberg"), [0.0, 0.5, 0.6, 1.0];
          max_cols=5,
          save_path=_output_path(output_dir, "heisenberg", "figures", "bond_energy_exact.json"))
  end
  ```

  Add the training-data target with an explicit raw input and processed output:

  ```julia
  function compute_heisenberg_training_history_data_target(; results_root=DEFAULT_RESULTS_ROOT, output_dir=results_root)
      source = _result_path(results_root, "heisenberg",
          "circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json")
      destination = _output_path(output_dir, "heisenberg", "figures",
          "circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json")
      compute_training_history_data(source; save_path=destination)
  end
  ```

  Change the bond plot target to pass `_result_path(results_root, "heisenberg", "figures", "bond_energy_exact.json")` to `plot_bond_energy_pattern_from_processed_data`. Change the training plot target to pass `_result_path(results_root, "heisenberg", "figures", "circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json")` to `plot_training_history_from_processed_data`; it may additionally pass static PEPSKit/DMRG reference paths, but may not pass a raw circuit path.

  Register both targets in `TARGETS`. Add each `data_target` and processed `source_data` path in `repro/figure_manifest.toml`. Rename every existing figure-level `source_data_sets` key to `compute_source_data_sets`; `repro/check.jl` already checks only `source_data` plus the old `source_data_sets` key, so the renamed raw lists remain explicit compute provenance without being presented as inputs to `plot`. Add the two source JSON filenames to the existing `project/results/heisenberg/figures -> results/heisenberg/processed` allowlist.

- [ ] **Step 4: Run the focused reproducibility test to verify it passes**

  Run: `julia --project=. test/runtests.jl reproducibility`

  Expected: all reproducibility assertions pass, including `plot` having no scheduled `data_target`s and both changed figures declaring processed sources.

### Task 4: Create canonical source data, rebuild staging, and update documentation

**Files:**
- Create: `project/results/heisenberg/figures/bond_energy_exact.json`
- Create: `project/results/heisenberg/figures/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json`
- Modify: `README.md:44-80`
- Modify: `REPRODUCIBILITY.md:50-98`
- Modify: `repro/stage_data.jl:137-148`
- Regenerate: `release-staging/IsoPEPS-paper-data/`

**Interfaces:**
- Consumes: Tasks 1-3 targets and `repro/stage_data.jl`.
- Produces: a staging package whose `processed/` directory contains both new canonical JSONs and whose manifest/checksums describe the final public workflow.

- [ ] **Step 1: Write the failing staged-package assertions**

  Add to `test/reproducibility.jl` a test that requires both staged canonical paths after staging:

  ```julia
  staged = joinpath(@__DIR__, "..", "release-staging", "IsoPEPS-paper-data")
  @test isfile(joinpath(staged, "results", "heisenberg", "processed", "bond_energy_exact.json"))
  @test isfile(joinpath(staged, "results", "heisenberg", "processed",
                         "circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json"))
  ```

- [ ] **Step 2: Run the focused test to verify it fails**

  Run: `julia --project=. test/runtests.jl reproducibility`

  Expected: failure because the current staging package does not yet contain the two JSON files.

- [ ] **Step 3: Generate, stage, and document the canonical files**

  Run the two new compute targets against the private source results to create the source JSONs. Update all prose so the primary command is:

  ```bash
  julia --project=. repro/reproduce.jl \
    --data-dir /path/to/IsoPEPS-paper-data \
    --output-dir reproduced-figures --mode plot
  ```

  Describe `compute` as validation-only and its `recomputed-data/` output. Update the staged README template with the same distinction.

  Rebuild staging in a fresh temporary destination first, run `repro/verify.jl` there, then replace the canonical `release-staging/IsoPEPS-paper-data/` only after the generated files and manifest are verified. The final staging command uses the private result root and manuscript image directory already used by the local release workflow.

- [ ] **Step 4: Run focused tests and staging verification**

  Run:

  ```bash
  julia --project=. test/runtests.jl reproducibility
  julia --project=. repro/verify.jl --data-dir release-staging/IsoPEPS-paper-data
  ```

  Expected: the test passes and `verify.jl` reports every staged file/checksum valid.

### Task 5: End-to-end primary-workflow verification

**Files:**
- No source edits required.

**Interfaces:**
- Consumes: final staging package and the default `repro/reproduce.jl` command.
- Produces: evidence that the user-facing workflow yields only final figures from canonical processed data.

- [ ] **Step 1: Render into a fresh temporary output directory without `--mode`**

  Run:

  ```bash
  output_dir=$(mktemp -d /private/tmp/ispeps-processed-figures.XXXXXX)
  julia --project=. repro/reproduce.jl \
    --data-dir release-staging/IsoPEPS-paper-data \
    --output-dir "$output_dir"
  ```

  Expected: the script reports `Rendered 17 paper figures from canonical processed data` and no raw contraction or resampling diagnostics appear.

- [ ] **Step 2: Verify output shape and manuscript coverage**

  Run:

  ```bash
  find "$output_dir" -type f -name '*.json' -print
  find "$output_dir" -maxdepth 1 -type f | wc -l
  julia --project=. repro/check.jl \
    --data-dir release-staging/IsoPEPS-paper-data \
    --tex /Users/rongyuqing/jcode/IsoPEPS-Notes/arxiv_submit/main.tex \
    --rendered-dir "$output_dir"
  ```

  Expected: no JSON path is printed, the count is `17`, and the checker reports `Checked 17 manuscript figures against main.tex.`

- [ ] **Step 3: Inspect the final diff without changing unrelated work**

  Run:

  ```bash
  git diff --check
  git status --short
  ```

  Expected: no whitespace errors. Preserve unrelated pre-existing modifications and report them rather than reverting them.
