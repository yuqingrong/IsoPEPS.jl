# IsoPEPS

[![Build Status](https://github.com/yuqingrong/IsoPEPS.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/yuqingrong/IsoPEPS.jl/actions/workflows/CI.yml?query=branch%3Amaster)

IsoPEPS.jl implements isometric projected entangled pair states on cylindrical
geometries using a spiral ordering, with optimization, exact and sampled
observables, and paper-figure post-processing.

## Install and test

Julia 1.10 or later is required.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Use the root package environment for package, test, and reproducibility commands.

## Reproduce the paper figures

First stage the private numerical results into a separate data package:

```bash
julia --project=. repro/stage_data.jl \
  --source-root /path/to/IsoPEPS.jl \
  --paper-image-dir /path/to/IsoPEPS-Notes/arxiv_submit/image \
  --destination release-staging/IsoPEPS-paper-data
```

The staged package separates saved circuit inputs in `raw/` from the canonical
figure-ready summaries in `processed/`.

Then redraw the paper figures from the canonical processed data:

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir reproduced-figures --mode plot
```

`--mode plot` is the recommended workflow. It redraws the figures from processed data without rerunning optimization, DMRG, or VUMPS calculations.

For validation, `--mode compute` recomputes declared processed-data targets from raw inputs, while `--mode archive` restores curated figure baselines.

See [the reproducibility guide](docs/reproducibility.md) for staging, validation, data layout, and release details.
