# Release-Staging Results Layout Design

## Goal

Make the staged paper-data package distinguish saved circuit inputs from
derived, figure-ready data without changing `project/results/` or numerical
values.

## Scope

- Change only the layout beneath `release-staging/IsoPEPS-paper-data/results/`
  and the release/reproduction metadata and code needed to consume it.
- Leave the working source tree at `project/results/` unchanged.
- Leave independent DMRG/VUMPS/iPEPS data under `results/reference/`.
- Treat only `circuit_*.json` files as model-local raw data. Sampling summaries
  such as M2 and readout-noise scans are processed data because they are
  already derived summaries rather than individual sample streams.

## Release Layout

```text
results/
  tfim_abc/
    raw/          circuit_*.json
    processed/    readout-noise and figure-ready JSONs
  heisenberg/
    raw/          circuit_*.json
    processed/    M2, structure-factor, and figure-ready JSONs
  reference/      DMRG/VUMPS/iPEPS reference scans
```

`processed/` is flat within each model. The existing `figures/` directory is
not retained in the staged package because every JSON in it is processed data.

## Data and Reproduction Flow

1. `repro/data_manifest.toml` will split each model's existing source directory
   into two destination entries: `raw/` for circuits and `processed/` for the
   remaining selected JSONs.
2. `repro/figure_manifest.toml` and its named scan data sets will point to the
   new staged paths.
3. `project/postprocess.jl` will receive a centralized read-path resolver. It
   will first support the legacy source layout, then map staged model inputs to
   `raw/` or `processed/`. This keeps development workflows unchanged while
   allowing `repro/reproduce.jl --mode compute` to run directly from the staged
   package.
4. Generated outputs remain in the existing output structure specified by
   `generated_output`; only inputs are reorganized.

## Validation

- Add a regression test for staged-model path resolution and the raw/processed
  classification.
- Build a fresh staging package and confirm no selected model JSON remains at
  the model root.
- Run the affected figure workflow, `repro/verify.jl`, `repro/check.jl`, and
  `git diff --check`.

## Non-Goals

- No new simulations, optimizations, DMRG/VUMPS calculations, or numerical
  changes.
- No moves inside `project/results/`.
- No reclassification of files under `results/reference/`.
