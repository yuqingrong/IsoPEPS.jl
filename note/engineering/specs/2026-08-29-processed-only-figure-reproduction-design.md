# Processed-Only Paper-Figure Reproduction Design

## Goal

Make the primary paper-figure workflow a strict, non-interactive
`canonical processed JSON -> manuscript figures` pipeline.  A reader running
`repro/reproduce.jl` in its default `plot` mode must not perform a tensor
contraction, resample a circuit, or write derived JSON data.

`archive` remains the byte-identical restoration path.  `compute` remains a
separate validation path that derives data from raw circuit inputs into an
explicit `recomputed-data/` output tree.

## Data products

The staged data package gains two additional canonical processed JSON files:

- `results/heisenberg/processed/bond_energy_exact.json`: the exact vertical,
  intra-cell horizontal, and inter-period horizontal bond expectations used by
  `bond_energy_exact.pdf`, with the model and circuit metadata for each of the
  four plotted `J2` values.
- `results/heisenberg/processed/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json`:
  the optimisation-history series and display metadata needed by the manuscript
  training-history figure.  It deliberately excludes the resampled expectation
  values, because that auxiliary plot is not a manuscript figure.

These files are canonical numeric inputs, like the existing TFIM and
Heisenberg `processed/*.json` products.  Their schema must retain the plotted
values and labels without requiring the raw optimisation-result object during
plotting.

## Modes and data flow

```text
raw circuits + references --compute--> recomputed-data/      (validation only)
canonical processed JSONs ----plot----> reproduced-figures/  (primary)
figures/published ---------archive--> reproduced-figures/    (exact baseline)
```

`plot` reads all generated numerical-figure inputs from
`results/*/processed/` and reference JSON files.  The two supplied
illustrations continue to be copied from `figures/published/`, since they are
not numerical plots.  The primary output directory contains only the 17
manuscript figure files; temporary renderer files are discarded.

`compute` writes regenerated JSON to `OUTPUT/recomputed-data/` and never
overwrites staged canonical processed files or produces manuscript figures.
It includes computation targets for the two new Heisenberg products as well as
the existing declared data targets.

## Code changes

1. Add processing and processed-data plotting functions for the exact
   bond-energy pattern and Heisenberg training history.  The plotting functions
   must accept a processed JSON path and have no raw-circuit, contraction, or
   sampling dependency.
2. Make `plot_analyze_heisenberg` use the processed training-history plotting
   function.  Remove its resampling and auxiliary expectation-values work from
   the reproduction path.
3. Make `plot_bond_energy_exact` use `bond_energy_exact.json` rather than raw
   circuit results.
4. Register new compute targets and update `figure_manifest.toml` so both
   figures declare their canonical processed JSON inputs and their validation
   data targets.  Raw circuit lists remain associated with `compute` only.
5. Add both files to the staged processed-data allowlist and regenerate the
   staging package only after the implementation is verified.
6. Update the root README, reproducibility guide, manifest command metadata,
   and staged-package README template to describe `plot` as processed-only.

## Errors and validation

- Processed-data plotting fails with a clear missing-file/schema error if a
  canonical processed JSON is absent or malformed.
- The existing checker continues to verify all manifest-declared source files
  and the 17 final output images.
- Tests cover the `plot`/`compute` split and ensure the two former raw-driven
  figure entries declare processed inputs plus compute targets.
- An end-to-end default `plot` run verifies that the output contains 17 final
  figures, no JSON files, no `recomputed-data/`, and no GUI display calls.
