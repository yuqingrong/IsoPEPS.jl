# Reproducibility and local release preparation

This repository is prepared for a future public release, but is not being
released now. No tag, GitHub release, Zenodo draft, DOI, upload, or manuscript
edit is part of this workflow.

## Code environment

Run all commands from the repository root with the tracked root environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The version intentionally remains `1.0.0-DEV`. `Manifest.toml` records the
dependency resolution used for this local preparation.

## Curated private data package

The source result directories are ignored and remain untouched. Build a new,
ignored staging package instead:

```bash
julia --project=. repro/stage_data.jl \
  --source-root /absolute/path/to/IsoPEPS.jl \
  --paper-image-dir /absolute/path/to/IsoPEPS-Notes/arxiv_submit/image \
  --destination release-staging/IsoPEPS-paper-data
```

The strict allowlist lives in `repro/data_manifest.toml`. It includes the
TFIM and Heisenberg results used by the paper, the DMRG/VUMPS/PEPSKit reference
data, and exact baselines copied from the manuscript image directory. It
excludes checkpoints, `.DS_Store`, and local reference states. The stager
compares SHA-256 values before and after copying, then writes
`SOURCE-MANIFEST.sha256` and `MANIFEST.sha256` in the staging package.

Validate the package before using it:

```bash
julia --project=. repro/verify.jl --data-dir release-staging/IsoPEPS-paper-data
```

## Figure reproduction

`repro/figure_manifest.toml` covers every `\includegraphics` item in the
read-only manuscript and identifies the source data, plotting target, exact
baseline, and baseline SHA-256.

Archive mode restores those exact files byte-for-byte. It is the default
because it is deterministic across CairoMakie and PDF-library versions:

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir release-staging/reproduced-figures --mode archive
```

Compute mode runs the corresponding plotting targets using only the staged
results and the precomputed bootstrap/variance intermediates. It writes to a
separate output directory and never reruns optimization, DMRG, or VUMPS:

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir release-staging/recomputed-figures --mode compute
```

For archive mode, require exact baseline hashes. For compute mode, the checker
verifies the full LaTex image list, checks all declared canonical data against
the package SHA-256 manifest, and rasterizes each generated PDF's first page
for a baseline comparison. The default mean pixel-error tolerance is `0.15` to
allow PDF metadata and renderer variation; pass `--raster-tolerance N` to make
that limit stricter.

```bash
julia --project=. repro/check.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --tex /absolute/path/to/IsoPEPS-Notes/arxiv_submit/main.tex \
  --rendered-dir release-staging/reproduced-figures --exact-baselines
```

## Deferred release checklist

The separate code and data metadata templates are
[`repro/zenodo-metadata-template.json`](repro/zenodo-metadata-template.json)
and [`repro/zenodo-data-metadata-template.json`](repro/zenodo-data-metadata-template.json).
See [`repro/RELEASE_CHECKLIST.md`](repro/RELEASE_CHECKLIST.md). When the
authors approve publication, create versioned Zenodo records rather than
altering an existing published record; Zenodo documents this model in its
[record versioning guide](https://help.zenodo.org/docs/deposit/about-records/).
