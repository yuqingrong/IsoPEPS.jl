# Contributing

Thanks for helping improve IsoPEPS. This repository mixes package code,
research scripts, and generated experiment outputs, so small workflow habits
make a big difference.

## Development Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
make init
```

Use the root project for package development. The `project/` directory has a
separate Julia environment for research scripts.

## Branches

- Base new work on `master`.
- Use descriptive branch names, for example `fix-transfer-spectrum`,
  `docs-quick-start`, or `feature-j1j2-observable`.
- Avoid reusing old experiment branches for unrelated work.
- Do not force-push shared branches unless everyone using that branch has
  agreed.

## Tests

Run the smallest relevant test group while developing:

```bash
make test-gates
make test-transfer_matrix
make test-observables
```

Before opening a pull request, run the full suite when practical:

```bash
make test
```

Named test groups are listed in `test/runtests.jl`.

## Formatting

Use Julia standard style with 4-space indentation. The repository includes a
JuliaFormatter configuration:

```bash
make format
make format-check
```

Keep formatting-only changes separate from behavioral changes when possible.

## Generated Files

Do not commit generated simulation outputs unless they are intentionally small
fixtures or documentation assets. The following paths are treated as local
outputs:

- `data/`
- `image/`
- `states/`
- `project/results/`
- `simulations/results/`

Small test fixtures should live under `test/data/`, be explicitly unignored
when needed, and be reviewed like source files.

## Pull Requests

Include:

- A short summary of the behavioral or documentation change.
- The test command you ran, or why a test was not practical.
- Any generated artifacts needed to review the change.
- Links to related issues, notes, or experiment records when available.

Prefer PR titles that explain the change, such as `Fix transfer spectrum for
2x2 unit cells`, instead of `update` or `up`.
