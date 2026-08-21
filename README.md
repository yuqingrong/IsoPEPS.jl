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

The root `Manifest.toml` is tracked so the dependency set used for a future
release is recorded. The package version remains `1.0.0-DEV` until author
approval for a public release.

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

To restore the exact curated figure baselines, use the deterministic archive
mode. To rerun the plotting targets from the staged raw results and precomputed
intermediates, choose compute mode; it never launches an optimization, DMRG,
or VUMPS calculation.

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir release-staging/reproduced-figures --mode archive

julia --project=. repro/check.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --tex /path/to/IsoPEPS-Notes/arxiv_submit/main.tex \
  --rendered-dir release-staging/reproduced-figures --exact-baselines
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete local workflow
and the deferred-public-release checklist.
