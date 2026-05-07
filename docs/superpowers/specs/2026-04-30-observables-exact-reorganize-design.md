# Design: Reorganize observables_exact.jl + Expand Test Coverage

Date: 2026-04-30

## Goal

1. Reorganize `src/observables_exact.jl` into a clean 8-section layout that matches the logical dependency order of the API.
2. Add comprehensive tests for every exported function that is currently untested.
3. Fix stale `3 * nqubits * p` param counts in existing `test/observables.jl` testsets.
4. Confirm `test/gates.jl` still passes (already improved in the previous session).

No logic changes — pure code movement and test additions.

---

## File Reorganization: `src/observables_exact.jl`

### New 8-section layout

| # | Section | Content |
|---|---------|---------|
| 1 | Internal Helpers | `_resolve_op`, `_fixed_points`, `_precompute_shifted_vectors`, `_dimer_cross_correlation` (moved from 5b), `_dimer_general_correlation` (moved from 5b) |
| 2 | Single-Site & Multi-Site Expectation Values | `expect(TransferOperator, obs)`, `expect(TransferOperator, sites::Dict)`, `expect` legacy wrappers, `_compute_pauli_total` (moved from §4), `compute_X_expectation` (moved from §4), `compute_Z_expectation` (moved from §4), `compute_single_expectation` (moved from §4) |
| 3 | Two-Point Correlation Functions | `correlation_function(TransferOperator, ...)`, `correlation_function(gates, ...)` legacy, `intercolumn_correlation` (moved from §6) |
| 4 | Bond Observables | `bond_expectation` (moved from §5b), `all_bond_expectations` (moved from §5b) |
| 5 | TFIM Energy | `compute_exact_energy(TFIM, TransferOperator)`, `compute_exact_energy(gates, ...)` legacy, `compute_ZZ_expectation` (stays here — closely tied to TFIM) |
| 6 | Heisenberg J1-J2 Energy | `compute_exact_heisenberg_energy(TransferOperator, J1, J2)`, two legacy wrappers (1×1, 2×2) |
| 7 | Spin-Spin / Dimer / Plaquette Correlations | `spin_spin_correlation`, `dimer_dimer_correlation`, `plaquette_plaquette_correlation` |
| 8 | Structure Factors | `spin_spin_structure_factor`, `magnetic_order_squared`, `dimer_structure_factor`, `plaquette_structure_factor` |

### What moves where

- `_dimer_cross_correlation`, `_dimer_general_correlation`: §5b → §1 (private helpers belong at top)
- `_compute_pauli_total`, `compute_X_expectation`, `compute_Z_expectation`, `compute_single_expectation`: §4 → §2 (expectation values, not energy)
- `intercolumn_correlation`: §6 → §3 (it is a correlation function)
- `bond_expectation`, `all_bond_expectations`: §5b → §4 (bond observables are their own category)

### What does NOT change

- Function signatures, docstrings, logic — zero behavior change
- Export list in `src/IsoPEPS.jl`

---

## Test Additions: `test/observables.jl`

### Fixes to existing tests

Replace `3 * nqubits * p` with `2 * nqubits * p` (= `IsoPEPS.PARAMS_PER_QUBIT_PER_LAYER * nqubits * p`) in all energy testsets where it is used as a param count.

### New testsets

**§2: expect() modern API**
- `expect_single_site`: `expect(op, :X)`, `expect(op, :Y)`, `expect(op, :Z)` on random gates — verify result is real, in [-1,1] per site, matches `compute_X_expectation` / `compute_Z_expectation` for scalar totals
- `expect_multi_site`: `expect(op, Dict((1,1)=>:Z, (1,2)=>:Z))` — verify real, in [-1,1]

**§3: correlation_function modern API**
- `correlation_function_transop`: `correlation_function(op, :Z, 1:5)` on random gates — verify each value is real, |val|≤1, and magnitude non-increasing on average (decay)

**§4: Bond observables**
- `bond_expectation_basic`: vertical and horizontal bonds on 1×1 and 2×2 unit cells — verify ∈ [-0.75, 0.75] (S·S ≤ 3/4), real
- `all_bond_expectations_shape`: verify returned arrays have shape `(row, N_cols)` for vert and `(row, N_cols-1)` for horiz; values in bounds

**§5: ZZ consistency**
- `compute_ZZ_expectation_consistency`: verify `ZZ_vert` matches the manual sum of `expect(gates, row, vq, Dict(i=>:Z, j=>:Z))` over all vertical bonds

**§6: Heisenberg energy**
- `compute_exact_heisenberg_energy_basic`: 1×1 unit cell, J2=0.0 — verify `Float64`, finite, negative for random params
- `heisenberg_2x2_basic`: 2×2 unit cell via `compute_exact_heisenberg_energy_2x2` — verify type and finiteness
- `heisenberg_energy_J2_sensitivity`: energy changes when J2 changes from 0 to 0.5

**§7: Multi-period correlations**
- `spin_spin_correlation_basic`: seps 0:4, verify real, |val|≤0.75, `corrs[0]` ≈ 0.75 for identical spins (S·S = 3/4)
- `dimer_dimer_correlation_vertical`: `connected=true` and `connected=false`, verify real and finite
- `dimer_dimer_correlation_horizontal`: same, with `dimer_orientation=:horizontal`
- `plaquette_plaquette_correlation_basic`: row≥2, 2-col unit cell, seps 0:3, verify real and finite

**§8: Structure factors (smoke tests)**
- `spin_spin_structure_factor_smoke`: q=(π,π) and q=(0,0), `max_separation=3` — verify `Float64`, finite, ≥0
- `dimer_structure_factor_smoke`: `:vertical` and `:horizontal`, `max_separation=3`
- `plaquette_structure_factor_smoke`: `max_separation=3`, row≥2, 2-col unit cell
- `magnetic_order_squared_smoke`: q=(π,π), verify ≥0

---

## Success Criteria

- All existing tests continue to pass (no behavior change)
- All new testsets pass
- CI passes on both Julia 1.11 and `pre`
