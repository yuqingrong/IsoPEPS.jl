# Release-Staging Results Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the staged TFIM and Heisenberg data into raw saved circuits and flat processed figure data, while leaving `project/results/` and every numerical value unchanged.

**Architecture:** A new, dependency-free resolver distinguishes the legacy development layout from the staged layout. `project/postprocess.jl` delegates all input resolution to it; the stage manifest uses separate source roots for `figures/` so the existing copier places processed JSON files directly in flat `processed/` directories.

**Tech Stack:** Julia 1.10+, TOML, Julia `Test`, existing reproduction scripts, CairoMakie post-processing.

**Spec:** `note/engineering/specs/2026-08-29-release-staging-results-layout-design.md`

## Global Constraints

- Change only the layout below `release-staging/IsoPEPS-paper-data/results/` and the code, manifests, tests, and documentation required to consume it.
- Do not move, edit, or recompute anything in `project/results/`.
- Preserve `results/tfim_abc/`, preserve all existing `generated_output` paths, and leave all `results/reference/` files unchanged.
- Only `circuit_*.json` is raw. M2, structure-factor, readout, variance, and figure-ready JSON files are processed.
- Each staged model's `processed/` directory is flat; never create `processed/figures/`.
- Build and validate a fresh temporary package first, then replace the canonical staging directory with no backup, as the user approved.
- Do not commit: this shared checkout is dirty and the user has not asked for a commit.

---

## File Structure

- `repro/results_layout.jl`: dependency-free path resolver.
- `project/postprocess.jl`: imports the resolver and retains its stable `_result_path` wrapper.
- `repro/data_manifest.toml`: split raw/processed selection entries.
- `repro/figure_manifest.toml`: declares the new staged source paths.
- `repro/stage_data.jl`: documents the split in the generated package README.
- `test/staged_results_layout.jl`: resolver regression test.
- `test/reproducibility.jl`: manifest classification and path assertions.
- `test/runtests.jl`: registers the resolver test.
- `README.md`, `REPRODUCIBILITY.md`: explain the staged-only distinction.

The final layout is:

```text
results/
  tfim_abc/
    raw/          21 circuit_tfim_*.json files
    processed/    7 TFIM figure JSONs + 1 readout JSON
  heisenberg/
    raw/          11 circuit_heisenberg_*.json files
    processed/    M2, structure-factor, and 2 figure JSONs
  reference/      unchanged DMRG/VUMPS/iPEPS scans
  mpskit_results_Ly=3_D=32.json
```

### Task 1: Define and test staged-result path resolution

**Files:**

- Create: `repro/results_layout.jl`
- Create: `test/staged_results_layout.jl`
- Modify: `test/runtests.jl:3-23`

**Interfaces:**

- Produces `ResultsLayout.result_path(results_root::AbstractString, pieces::AbstractString...)::String`.
- Inputs under an existing legacy source path return the requested legacy path.
- Staged TFIM/Heisenberg circuit paths return `raw/`; all other selected model-local paths return `processed/`; reference paths are never transformed.

- [ ] **Step 1: Write the failing resolver test**

Create `test/staged_results_layout.jl`:

```julia
include(joinpath(@__DIR__, "..", "repro", "results_layout.jl"))
using .ResultsLayout

@testset "staged and legacy result paths" begin
    mktempdir() do root
        legacy_circuit = joinpath(root, "tfim_abc", "circuit_tfim.json")
        mkpath(dirname(legacy_circuit))
        write(legacy_circuit, "legacy")
        @test result_path(root, "tfim_abc", "circuit_tfim.json") == legacy_circuit

        rm(joinpath(root, "tfim_abc"); recursive=true)
        for directory in (
            joinpath(root, "tfim_abc", "raw"),
            joinpath(root, "tfim_abc", "processed"),
            joinpath(root, "heisenberg", "raw"),
            joinpath(root, "heisenberg", "processed"),
            joinpath(root, "reference"),
        )
            mkpath(directory)
        end

        @test result_path(root, "tfim_abc") == joinpath(root, "tfim_abc", "raw")
        @test result_path(root, "tfim_abc", "circuit_tfim.json") ==
              joinpath(root, "tfim_abc", "raw", "circuit_tfim.json")
        @test result_path(root, "tfim_abc", "figures", "tfim_energy_vs_g.json") ==
              joinpath(root, "tfim_abc", "processed", "tfim_energy_vs_g.json")
        @test result_path(root, "tfim_abc", "readout_noise_energ_g=3.0.json") ==
              joinpath(root, "tfim_abc", "processed", "readout_noise_energ_g=3.0.json")
        @test result_path(root, "heisenberg") == joinpath(root, "heisenberg", "raw")
        @test result_path(root, "heisenberg", "circuit_heisenberg.json") ==
              joinpath(root, "heisenberg", "raw", "circuit_heisenberg.json")
        @test result_path(root, "heisenberg", "M2_sampling.json") ==
              joinpath(root, "heisenberg", "processed", "M2_sampling.json")
        @test result_path(root, "reference", "dmrg.json") == joinpath(root, "reference", "dmrg.json")
    end
end
```

Register it immediately before `reproducibility` in `TEST_FILES`:

```julia
"staged_results_layout" => "staged_results_layout.jl",
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl staged_results_layout
```

Expected: failure because `repro/results_layout.jl` does not exist.

- [ ] **Step 3: Implement the resolver**

Create `repro/results_layout.jl`:

```julia
module ResultsLayout

export result_path

const STAGED_MODELS = ("tfim_abc", "heisenberg")

function result_path(results_root::AbstractString, pieces::AbstractString...)::String
    requested = joinpath(results_root, pieces...)
    length(pieces) > 1 && (isfile(requested) || isdir(requested)) && return requested
    isempty(pieces) && return requested

    model = first(pieces)
    model in STAGED_MODELS || return requested
    model_root = joinpath(results_root, model)
    staged_raw = joinpath(model_root, "raw")
    isdir(staged_raw) || return requested

    length(pieces) == 1 && return staged_raw
    remaining = pieces[2:end]
    first(remaining) == "figures" && return joinpath(model_root, "processed", remaining[2:end]...)
    startswith(first(remaining), "circuit_") && return joinpath(staged_raw, remaining...)
    return joinpath(model_root, "processed", remaining...)
end

end
```

- [ ] **Step 4: Run the resolver test to verify it passes**

Run:

```bash
julia --project=. test/runtests.jl staged_results_layout
```

Expected: PASS with nine assertions.

- [ ] **Step 5: Review without committing**

Run:

```bash
git diff -- repro/results_layout.jl test/staged_results_layout.jl test/runtests.jl
```

Expected: only the new resolver, its regression test, and test registration are visible. Leave them uncommitted.

### Task 2: Integrate post-processing with the resolver

**Files:**

- Modify: `project/postprocess.jl:1-7`
- Modify: `project/postprocess.jl:133-136`
- Modify: `project/postprocess.jl:190-194`
- Modify: `test/staged_results_layout.jl`

**Interfaces:**

- Consumes `ResultsLayout.result_path` from Task 1.
- Retains `_result_path(results_root::AbstractString, pieces...)::String` for every existing post-processing caller.
- Keeps `_output_path` as the only helper for generated data/output paths.

- [ ] **Step 1: Extend the failing test with source-level integration checks**

Append to `test/staged_results_layout.jl`:

```julia
@testset "postprocess result-path delegation" begin
    source = read(joinpath(@__DIR__, "..", "project", "postprocess.jl"), String)
    @test occursin("include(joinpath(@__DIR__, \"..\", \"repro\", \"results_layout.jl\"))", source)
    @test occursin("_result_path(results_root::AbstractString, pieces...) = result_path(results_root, pieces...)", source)
    @test occursin("data_file=_output_path(output_dir, \"heisenberg\", \"sf_discrete.json\")", source)
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl staged_results_layout
```

Expected: failure because the postprocessor has not imported or used the resolver.

- [ ] **Step 3: Make the minimal integration changes**

At the beginning of `project/postprocess.jl`, immediately before `using IsoPEPS`, add:

```julia
include(joinpath(@__DIR__, "..", "repro", "results_layout.jl"))
using .ResultsLayout: result_path
```

Replace line 135 with:

```julia
_result_path(results_root::AbstractString, pieces...) = result_path(results_root, pieces...)
```

In `plot_structure_factors_combined_discrete`, replace:

```julia
data_file=_result_path(output_dir, "heisenberg", "sf_discrete.json")
```

with:

```julia
data_file=_output_path(output_dir, "heisenberg", "sf_discrete.json")
```

This avoids interpreting a generated output directory as an input package.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
julia --project=. test/runtests.jl staged_results_layout
```

Expected: PASS. The test remains free of CairoMakie imports.

- [ ] **Step 5: Review without committing**

Run:

```bash
git diff -- project/postprocess.jl repro/results_layout.jl test/staged_results_layout.jl
```

Expected: legacy source-tree call sites remain unchanged and output locations remain under the existing generated hierarchy.

### Task 3: Split the staging allowlist into raw and flat processed inputs

**Files:**

- Modify: `repro/data_manifest.toml:13-74`
- Modify: `repro/stage_data.jl:117-128`
- Modify: `test/reproducibility.jl:13-28`

**Interfaces:**

- Consumes the existing `source`, `destination`, `purpose`, and `include_files` manifest fields.
- Produces five TFIM/Heisenberg directory entries, one TFIM processed file entry, and the unchanged reference entry.
- Does not modify `copy_selected_tree!`; source directory selection achieves flattening.

- [ ] **Step 1: Replace the old one-directory-per-model test with failing raw/processed classification tests**

In `test/reproducibility.jl`, replace the current directory-source equality assertion and the Heisenberg `only(filter(...))` lookup with:

```julia
model_entries = filter(entry -> startswith(entry["destination"], "results/tfim_abc/") ||
                                startswith(entry["destination"], "results/heisenberg/"),
                       data_manifest["directories"])
@test [entry["destination"] for entry in model_entries] == [
    "results/tfim_abc/raw",
    "results/tfim_abc/processed",
    "results/heisenberg/raw",
    "results/heisenberg/processed",
    "results/heisenberg/processed",
]
raw_entries = filter(entry -> endswith(entry["destination"], "/raw"), model_entries)
@test all(entry -> all(name -> startswith(name, "circuit_"), entry["include_files"]), raw_entries)
processed_entries = filter(entry -> endswith(entry["destination"], "/processed"), model_entries)
@test all(entry -> all(name -> !startswith(name, "circuit_"), entry["include_files"]), processed_entries)
@test all(entry -> !occursin("figures/", entry["destination"]), processed_entries)
heisenberg_raw = only(filter(entry -> entry["destination"] == "results/heisenberg/raw", model_entries))["include_files"]
@test !any(filename -> occursin("J2=0.51", filename) || occursin("nqubits=5", filename), heisenberg_raw)
```

Also assert the separate TFIM readout entry:

```julia
tfim_readout = only(filter(entry -> entry["destination"] == "results/tfim_abc/processed/readout_noise_energ_g=3.0.json",
                           data_manifest["files"]))
@test tfim_readout["source"] == "project/results/tfim_abc/readout_noise_energ_g=3.0.json"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl reproducibility
```

Expected: failure because current model destinations are `results/tfim_abc` and `results/heisenberg`.

- [ ] **Step 3: Replace the TFIM and Heisenberg allowlist entries**

Use exactly these TFIM processed entries:

```toml
[[directories]]
source = "project/results/tfim_abc/figures"
destination = "results/tfim_abc/processed"
purpose = "TFIM derived figure-ready data"
include_files = [
    "tfim_energy_vs_g.json",
    "tfim_energy_vs_g_error.json",
    "energy_dynamics_vs_g.json",
    "tfim_correlation_length_vs_g.json",
    "tfim_connected_correlation_vs_g.json",
    "tfim_magnetization_vs_g.json",
    "tfim_variance_vs_samples.json",
]

[[files]]
source = "project/results/tfim_abc/readout_noise_energ_g=3.0.json"
destination = "results/tfim_abc/processed/readout_noise_energ_g=3.0.json"
purpose = "TFIM derived readout-noise scan"
```

Use exactly these Heisenberg processed entries:

```toml
[[directories]]
source = "project/results/heisenberg/figures"
destination = "results/heisenberg/processed"
purpose = "J1-J2 derived figure-ready data"
include_files = [
    "heisenberg_energy_vs_J2.json",
    "heisenberg_variance_vs_samples.json",
]

[[directories]]
source = "project/results/heisenberg"
destination = "results/heisenberg/processed"
purpose = "J1-J2 derived sampling summaries"
include_files = [
    "M2_sampling.json",
    "sf.json",
]
```

Use this complete TFIM raw list unchanged except for its destination and purpose:

```toml
[[directories]]
source = "project/results/tfim_abc"
destination = "results/tfim_abc/raw"
purpose = "TFIM optimized circuit inputs used by manuscript figure workflows"
include_files = [
    "circuit_tfim_J=1.0_g=0.0_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=0.25_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=0.5_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=0.75_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=1.0_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=1.25_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=1.5_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=1.75_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=2.0_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=2.25_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=2.5_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=2.75_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=3.0_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=3.25_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=3.5_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=3.75_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=4.0_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=4.25_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=4.5_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=4.75_row=3_p=3_nqubits=3_1x3.json",
    "circuit_tfim_J=1.0_g=5.0_row=3_p=3_nqubits=3_1x3.json",
]
```

Use this complete Heisenberg raw list unchanged except for its destination and purpose:

```toml
[[directories]]
source = "project/results/heisenberg"
destination = "results/heisenberg/raw"
purpose = "J1-J2 optimized circuit inputs used by manuscript figure workflows"
include_files = [
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.0_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.1_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.2_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.3_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.4_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.6_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.7_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.8_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=0.9_row=4_p=3_nqubits=3_2x2.json",
    "circuit_heisenberg_j1j2_J1=1.0_J2=1.0_row=4_p=3_nqubits=3_2x2.json",
]
```

Leave the existing reference directory entry and the root `mpskit_results_Ly=3_D=32.json` file entry unchanged. In `write_package_readme`, replace the first `results/` bullet with:

```markdown
- `results/tfim_abc/raw/` and `results/heisenberg/raw/` contain saved
  optimized circuit inputs. Their sibling `processed/` directories contain
  derived figure-ready and sampling-summary JSONs. `results/reference/`
  contains the DMRG/VUMPS/PEPSKit references required by the paper figures.
```

- [ ] **Step 4: Run tests and build a temporary package**

Run:

```bash
julia --project=. test/runtests.jl reproducibility
julia --project=. repro/stage_data.jl \
  --source-root /Users/rongyuqing/jcode/IsoPEPS.jl \
  --paper-image-dir /Users/rongyuqing/jcode/IsoPEPS-Notes/arxiv_submit/image \
  --destination /private/tmp/ispeps-paper-data-layout-check
find /private/tmp/ispeps-paper-data-layout-check/results/tfim_abc -maxdepth 2 -type f | sort
find /private/tmp/ispeps-paper-data-layout-check/results/heisenberg -maxdepth 2 -type f | sort
```

Expected: the test passes; circuit JSONs occur only below `raw/`; selected non-circuit model JSONs occur directly below `processed/`; no selected JSON occurs at a model root.

- [ ] **Step 5: Review without committing**

Run:

```bash
git diff -- repro/data_manifest.toml repro/stage_data.jl test/reproducibility.jl
```

Expected: no source JSON files have been moved or altered.

### Task 4: Declare the relocated staged inputs in the figure manifest

**Files:**

- Modify: `repro/figure_manifest.toml:10-227`
- Modify: `test/reproducibility.jl:29-39`

**Interfaces:**

- Consumes the raw/processed staged layout from Task 3.
- Produces source data that `repro/check.jl` can find in a fresh staged package.
- Leaves plot targets, `generated_output`, baseline filenames, and SHA-256 baseline values unchanged.

- [ ] **Step 1: Add failing path assertions**

After parsing `figure_manifest` in `test/reproducibility.jl`, add:

```julia
@test all(path -> startswith(path, "results/tfim_abc/raw/circuit_tfim_"),
          figure_manifest["data_sets"]["tfim_g_scan"])
@test all(path -> startswith(path, "results/tfim_abc/raw/circuit_tfim_"),
          figure_manifest["data_sets"]["tfim_dynamics_scan"])
@test all(path -> startswith(path, "results/heisenberg/raw/circuit_heisenberg_"),
          figure_manifest["data_sets"]["heisenberg_j2_scan"])
@test all(path -> startswith(path, "results/heisenberg/raw/circuit_heisenberg_"),
          figure_manifest["data_sets"]["bond_energy_scan"])
declared_model_sources = String[
    source for figure in figures for source in get(figure, "source_data", String[])
    if startswith(source, "results/tfim_abc/") || startswith(source, "results/heisenberg/")
]
@test all(path -> occursin(r"results/(tfim_abc|heisenberg)/(raw|processed)/", path),
          declared_model_sources)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl reproducibility
```

Expected: failure because the current paths point at model roots and `figures/`.

- [ ] **Step 3: Change only model-local source paths**

Apply these exact replacements to `repro/figure_manifest.toml`:

```text
results/tfim_abc/circuit_                    -> results/tfim_abc/raw/circuit_
results/heisenberg/circuit_                  -> results/heisenberg/raw/circuit_
results/tfim_abc/figures/                    -> results/tfim_abc/processed/
results/heisenberg/figures/                  -> results/heisenberg/processed/
results/tfim_abc/readout_noise_energ_g=3.0.json
                                               -> results/tfim_abc/processed/readout_noise_energ_g=3.0.json
results/heisenberg/M2_sampling.json           -> results/heisenberg/processed/M2_sampling.json
results/heisenberg/sf.json                    -> results/heisenberg/processed/sf.json
```

Do not modify reference paths, the root MPSKit result path, targets, output paths, baselines, or hashes.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run:

```bash
julia --project=. test/runtests.jl staged_results_layout reproducibility
```

Expected: PASS.

- [ ] **Step 5: Review without committing**

Run:

```bash
git diff -- repro/figure_manifest.toml test/reproducibility.jl
```

Expected: every model-local figure input contains exactly one `raw/` or `processed/` segment.

### Task 5: Document the staged-only layout

**Files:**

- Modify: `README.md:26-57`
- Modify: `REPRODUCIBILITY.md:18-61`
- Modify: `test/reproducibility.jl`

**Interfaces:**

- Consumes the layout in Task 3.
- Produces matching user-facing instructions without changing any commands.

- [ ] **Step 1: Add a failing documentation test**

Add to the existing reproducibility testset:

```julia
for documentation in ("README.md", "REPRODUCIBILITY.md")
    text = read(joinpath(@__DIR__, "..", documentation), String)
    @test occursin("raw/", text)
    @test occursin("processed/", text)
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl reproducibility
```

Expected: failure because the documents do not yet explain the split.

- [ ] **Step 3: Add this paragraph after each document's staging-allowlist description**

```markdown
Within the staged package only, `results/tfim_abc/` and
`results/heisenberg/` separate saved `circuit_*.json` inputs in `raw/`
from derived sampling summaries and figure-ready JSONs in `processed/`.
`results/reference/` remains separate for independent DMRG, VUMPS, and iPEPS
reference scans. The private `project/results/` source layout is not changed.
```

- [ ] **Step 4: Run tests and formatting validation**

Run:

```bash
julia --project=. test/runtests.jl reproducibility
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 5: Review without committing**

Run:

```bash
git diff -- README.md REPRODUCIBILITY.md test/reproducibility.jl
```

Expected: documents describe only staging and never claim that source data moved.

### Task 6: Rebuild, verify, reproduce, and replace the canonical staging package

**Files:**

- Create temporarily: `/private/tmp/ispeps-paper-data-final`
- Create temporarily: `/private/tmp/ispeps-paper-data-rendered`
- Replace: `release-staging/IsoPEPS-paper-data`

**Interfaces:**

- Consumes all updated code and manifests.
- Produces a canonical private staging package with the approved split.
- Uses the existing `stage_data.jl`, `verify.jl`, `check.jl`, and `reproduce.jl` interfaces.

- [ ] **Step 1: Run all relevant tests before staging data**

Run:

```bash
julia --project=. test/runtests.jl staged_results_layout reproducibility
```

Expected: PASS.

- [ ] **Step 2: Build and verify a fresh temporary staging package**

Run:

```bash
rm -rf /private/tmp/ispeps-paper-data-final /private/tmp/ispeps-paper-data-rendered
julia --project=. repro/stage_data.jl \
  --source-root /Users/rongyuqing/jcode/IsoPEPS.jl \
  --paper-image-dir /Users/rongyuqing/jcode/IsoPEPS-Notes/arxiv_submit/image \
  --destination /private/tmp/ispeps-paper-data-final
julia --project=. repro/verify.jl \
  --data-dir /private/tmp/ispeps-paper-data-final
```

Expected: the stager reports unchanged source checksums and the verifier accepts the new package.

- [ ] **Step 3: Validate layout and declared source data**

Run:

```bash
find /private/tmp/ispeps-paper-data-final/results/tfim_abc -maxdepth 1 -type f
find /private/tmp/ispeps-paper-data-final/results/heisenberg -maxdepth 1 -type f
julia --project=. repro/check.jl \
  --data-dir /private/tmp/ispeps-paper-data-final \
  --tex /Users/rongyuqing/jcode/IsoPEPS-Notes/arxiv_submit/main.tex
```

Expected: the two `find` commands print no model-root JSON files, and `check.jl` locates every declared raw, processed, and reference file.

- [ ] **Step 4: Compute all staged figures and check the render results**

Run:

```bash
julia --project=. repro/reproduce.jl \
  --data-dir /private/tmp/ispeps-paper-data-final \
  --output-dir /private/tmp/ispeps-paper-data-rendered --mode compute
julia --project=. repro/check.jl \
  --data-dir /private/tmp/ispeps-paper-data-final \
  --tex /Users/rongyuqing/jcode/IsoPEPS-Notes/arxiv_submit/main.tex \
  --rendered-dir /private/tmp/ispeps-paper-data-rendered
```

Expected: compute mode uses staged input data and produces all 17 figure files; the checker accepts the supplied images and generated PDFs under the existing raster tolerance.

- [ ] **Step 5: Replace canonical staging with no backup after all validation passes**

Run only after Steps 1-4 pass:

```bash
rm -rf /Users/rongyuqing/jcode/IsoPEPS.jl/release-staging/IsoPEPS-paper-data
mv /private/tmp/ispeps-paper-data-final /Users/rongyuqing/jcode/IsoPEPS.jl/release-staging/IsoPEPS-paper-data
julia --project=. repro/verify.jl \
  --data-dir /Users/rongyuqing/jcode/IsoPEPS.jl/release-staging/IsoPEPS-paper-data
git diff --check
```

Expected: canonical staging has the new raw/processed layout, no backup is retained, verification passes, and no whitespace errors exist. Keep every source change uncommitted.
