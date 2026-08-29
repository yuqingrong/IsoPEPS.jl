# TFIM Energy Overlap Markers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render coincident IsoPEPS and DMRG data points as a visible blue filled circle framed by an orange open square in the TFIM energy figure.

**Architecture:** Keep `plot_energy_error_vs_g` as the sole plotting entry point. Its existing circuit loop keeps the filled-circle styling, while DMRG references in its reference loop receive transparent marker fill, an orange stroke, and a larger square. The existing plotting handles supply the legend appearance automatically.

**Tech Stack:** Julia 1.10, CairoMakie 0.15, Test, Poppler PDF rendering.

## Global Constraints

- Change only the energy-comparison rendering path and its focused visualization test.
- Preserve all energy data, axes, labels, legend placement, and DMRG dotted line styling.
- Apply the open-square treatment only when `startswith(series.label, "DMRG")` is true.
- Regenerate `project/results/tfim_abc/figures/tfim_energy_vs_g.pdf` through `project/postprocess.jl` target `energy-vs-g-tfim`.

---

### Task 1: Test and implement open DMRG reference markers

**Files:**

- Modify: `test/visualization.jl:402-430`
- Modify: `src/visualization/scan_comparisons.jl:596-605`

**Interfaces:**

- Consumes: `plot_energy_error_vs_g(data_dir::String, scan_values::Vector{Float64}; dmrg_file, circuit_series, energy_source)`.
- Produces: a `Figure` whose DMRG `ScatterLines` plots retain `marker=:rect` but have `markercolor=:transparent`, `strokecolor` matching the series `color`, positive `strokewidth`, and a marker size larger than the IsoPEPS circle.

- [ ] **Step 1: Write the failing test**

  In the existing `plot_energy_error_vs_g supports multiple circuit and reference series` test, inspect `fig_energy`'s axis plots after the existing data assertions:

  ```julia
  energy_axis = only(filter(content -> content isa Axis, fig_energy.content))
  energy_series = filter(plot -> hasproperty(plot, :marker) &&
                                hasproperty(plot, :markercolor) &&
                                hasproperty(plot, :markersize),
                         energy_axis.scene.plots)
  isopeps_plot = only(filter(plot -> plot.marker[] == Makie.to_spritemarker(:circle),
                             energy_series))
  dmrg_plots = filter(plot -> plot.marker[] == Makie.to_spritemarker(:rect),
                      energy_series)
  @test length(dmrg_plots) == 2
  @test all(plot -> plot.markercolor[] == :transparent, dmrg_plots)
  @test all(plot -> plot.strokecolor[] == plot.color[], dmrg_plots)
  @test all(plot -> plot.strokewidth[] > 0, dmrg_plots)
  @test all(plot -> plot.markersize[] > isopeps_plot.markersize[], dmrg_plots)
  ```

- [ ] **Step 2: Run the focused test to verify it fails**

  Run:

  ```bash
  julia --project=. -e 'using IsoPEPS, Test; include("test/visualization.jl")'
  ```

  Expected: the new marker-color assertion fails because current DMRG squares are filled with their series color.

- [ ] **Step 3: Write the minimal implementation**

  In the reference loop in `plot_energy_error_vs_g`, calculate `is_dmrg = startswith(series.label, "DMRG")`, then retain the existing color, square marker, and dotted line while passing DMRG-only marker keyword arguments:

  ```julia
  is_dmrg = startswith(series.label, "DMRG")
  ref_marker = is_dmrg ? :rect : markers[mod1(idx + length(circuits), length(markers))]
  ref_marker_size = is_dmrg ? 1.25 * markersize : markersize
  ref_marker_attrs = is_dmrg ?
      (; markercolor=:transparent, strokecolor=ref_color, strokewidth=1.2) :
      NamedTuple()
  scatterlines!(ax1, series.scan_values, series.energies;
                label=_energy_plot_label(series.label, is_heisenberg),
                color=ref_color,
                marker=ref_marker,
                markersize=ref_marker_size,
                linestyle=ref_style,
                ref_marker_attrs...)
  ```

- [ ] **Step 4: Run the focused test to verify it passes**

  Run:

  ```bash
  julia --project=. -e 'using IsoPEPS, Test; include("test/visualization.jl")'
  ```

  Expected: all visualization tests pass, including the DMRG open-marker assertions.

- [ ] **Step 5: Commit the implementation**

  ```bash
  git add src/visualization/scan_comparisons.jl test/visualization.jl
  git commit -m "fix: reveal overlapping IsoPEPS energy markers"
  ```

### Task 2: Regenerate and visually validate the requested PDF

**Files:**

- Regenerate: `project/results/tfim_abc/figures/tfim_energy_vs_g.pdf`

**Interfaces:**

- Consumes: `TARGETS["energy-vs-g-tfim"]` from `project/postprocess.jl`.
- Produces: the requested one-page PDF rendered at final figure geometry with both blue IsoPEPS circles and orange open DMRG squares visible.

- [ ] **Step 1: Regenerate the figure**

  Run:

  ```bash
  julia --project=. project/postprocess.jl energy-vs-g-tfim
  ```

  Expected: output reports `Energy figure saved to: project/results/tfim_abc/figures/tfim_energy_vs_g.pdf`.

- [ ] **Step 2: Render the PDF for visual QA**

  Run:

  ```bash
  mkdir -p tmp/pdfs
  pdftoppm -png -r 180 project/results/tfim_abc/figures/tfim_energy_vs_g.pdf tmp/pdfs/tfim-energy-overlap
  ```

  Inspect `tmp/pdfs/tfim-energy-overlap-1.png` at original resolution.

- [ ] **Step 3: Verify final output**

  Confirm the blue IsoPEPS circles are visible within the orange open DMRG squares at overlapping points; confirm axes, labels, and legend remain unclipped and legible.

- [ ] **Step 4: Commit the regenerated artifact if it is versioned**

  ```bash
  git add project/results/tfim_abc/figures/tfim_energy_vs_g.pdf
  git commit -m "docs: refresh TFIM energy comparison figure"
  ```
