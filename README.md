# IsoPEPS

[![Build Status](https://github.com/yuqingrong/IsoPEPS.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/yuqingrong/IsoPEPS.jl/actions/workflows/CI.yml?query=branch%3Amaster)

IsoPEPS.jl implements isometric projected entangled pair states for cylinder
geometries, including optimization workflows, exact and sampled observables,
and paper-figure post-processing.

## Local setup

Use the tracked root environment for every package, test, and reproducibility
command. Julia 1.10 or later is required by `Project.toml`.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The root `Manifest.toml` records the dependency resolution for the `1.0.0`
release candidate. It becomes a public release only after final author
approval, a reviewed merge to `master`, and a version tag.

The `project/` directory contains research scripts, not a second supported
environment. Its historical `Project.toml` and `Manifest.toml` remain only for
legacy reference; run those scripts with the root environment.

## Local paper-data staging and figures

The numerical result tree remains private and ignored by Git. To make a
separate local package for a future data release, point the staging command at
the private source checkout and the manuscript's existing image directory:

```bash
julia --project=. repro/stage_data.jl \
  --source-root /path/to/IsoPEPS.jl \
  --paper-image-dir /path/to/IsoPEPS-Notes/arxiv_submit/image \
  --destination release-staging/IsoPEPS-paper-data

julia --project=. repro/verify.jl \
  --data-dir release-staging/IsoPEPS-paper-data
```

The staging tool copies only the allowlist in
[`repro/data_manifest.toml`](repro/data_manifest.toml), excludes checkpoints,
local states, and desktop metadata, and verifies source checksums before and
after copying. It does not change the source results or create a Zenodo record.

Within the staged package only, `results/tfim_abc/` and `results/heisenberg/`
separate saved `circuit_*.json` inputs in `raw/` from derived sampling
summaries and figure-ready JSONs in `processed/`. `results/reference/` remains
separate for independent DMRG, VUMPS, and iPEPS reference scans. The private
`project/results/` source layout is not changed.

The primary reproduction command is plot mode. It reads the canonical
`processed/*.json` figure data in the staged package and writes the 17 paper
figures directly to the requested output directory; it does not regenerate or
copy processed JSON files. This is the command readers should use to redraw the
paper figures. It never launches an optimization, DMRG, or VUMPS calculation.

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir reproduced-figures --mode plot

julia --project=. repro/check.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --tex /path/to/IsoPEPS-Notes/arxiv_submit/main.tex \
  --rendered-dir reproduced-figures
```

Archive mode remains available to restore the exact curated PDF/PNG baselines
byte-for-byte. Compute mode is a separate validation workflow: it recalculates
declared data targets from raw inputs into `recomputed-data/` and does not
replace the canonical processed data or write the paper figures.

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir reproduced-figures --mode archive

julia --project=. repro/check.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --tex /path/to/IsoPEPS-Notes/arxiv_submit/main.tex \
  --rendered-dir reproduced-figures --exact-baselines
```

See [the reproducibility guide](docs/reproducibility.md) for the complete local workflow
and the deferred-public-release checklist.
