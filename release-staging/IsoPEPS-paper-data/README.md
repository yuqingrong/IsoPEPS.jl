# IsoPEPS paper data package (local staging copy)

This directory is a private, local staging package for the IsoPEPS Notes
paper. It was produced by `repro/stage_data.jl` from the strict allowlist in
`data_manifest.toml`; it is not a public release or a Zenodo deposit.

- `results/` contains raw optimization results, precomputed bootstrap and
  variance intermediates, and the DMRG/VUMPS/PEPSKit references required by
  the paper figures.
- `figures/published/` contains the exact figure files used by the manuscript.
- `figure_manifest.toml` maps every manuscript figure to its inputs, command,
  and baseline checksum.
- `MANIFEST.sha256` verifies every file in this package.

The data are intended for future release under CC BY 4.0; see
`LICENSE-CC-BY-4.0.txt`. Until the authors approve a release, keep this
directory private.
