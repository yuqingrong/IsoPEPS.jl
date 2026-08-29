# Reproducibility and local release preparation

This repository contains the `1.0.0` release candidate, but that version is
not public until it is approved, merged, and tagged. No tag, GitHub release,
Zenodo draft, DOI, upload, or manuscript edit is part of this workflow.

## Code environment

Run all commands from the repository root with the root package environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

`Project.toml` and `CITATION.cff` declare version `1.0.0`. The root package
does not track a `Manifest.toml`, so CI can resolve dependencies compatible
with Julia 1.10, 1.11, and pre-release Julia.

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

Within the staged package only, `results/tfim_abc/` and `results/heisenberg/`
separate saved `circuit_*.json` inputs in `raw/` from derived sampling
summaries and figure-ready JSONs in `processed/`. `results/reference/` remains
separate for independent DMRG, VUMPS, and iPEPS reference scans. The private
`project/results/` source layout is not changed.

Validate the package before using it:

```bash
julia --project=. repro/verify.jl --data-dir release-staging/IsoPEPS-paper-data
```

## Figure reproduction

`repro/figure_manifest.toml` covers every `\includegraphics` item in the
read-only manuscript and identifies the source data, plotting target, exact
baseline, and baseline SHA-256.

Plot mode is the primary figure-reproduction workflow. It is the default and
uses the canonical `results/*/processed/*.json` files directly; it creates no
new processed JSON files. The output directory contains the 17 figures cited by
the manuscript.

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir reproduced-figures --mode plot
```

Archive mode restores the exact curated files byte-for-byte. Use it when an
identical copy of the submitted PDF/PNG assets is required:

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir reproduced-figures --mode archive
```

Compute mode is an optional raw-data validation workflow. It recomputes only
the declared processed-data targets into `recomputed-data/`; it neither changes
the staged canonical processed JSON files nor renders paper figures. It never
reruns optimization, DMRG, or VUMPS:

```bash
julia --project=. repro/reproduce.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --output-dir recomputation-check --mode compute
```

For plot mode, the checker verifies the full LaTex image list, checks all
declared canonical data against the package SHA-256 manifest, and rasterizes
each generated PDF's first page for a baseline comparison. The default mean
pixel-error tolerance is `0.15`; pass `--raster-tolerance N` to make that limit
stricter. For archive mode, add `--exact-baselines`.

```bash
julia --project=. repro/check.jl \
  --data-dir release-staging/IsoPEPS-paper-data \
  --tex /absolute/path/to/IsoPEPS-Notes/arxiv_submit/main.tex \
  --rendered-dir reproduced-figures
```

## Deferred release checklist

The repository-root [`.zenodo.json`](../.zenodo.json) supplies metadata when
Zenodo archives a GitHub software release. The separate
[`repro/zenodo-data-metadata-template.json`](../repro/zenodo-data-metadata-template.json)
is the metadata payload for the curated data package. See
[`repro/RELEASE_CHECKLIST.md`](../repro/RELEASE_CHECKLIST.md). When the authors
approve publication, create versioned Zenodo records rather than altering an
existing published record; Zenodo documents this model in its
[record versioning guide](https://help.zenodo.org/docs/deposit/about-records/).
