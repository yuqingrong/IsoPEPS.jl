# Strict iDMRG Bulk-Energy Convergence Design

## Goal

Compute the `J1 = 1`, `J2 = 0.5` Heisenberg-model energy density on the
`Ly = 4` infinite cylinder, increasing the MPS bond dimension until both the
state solver and the raw consecutive-bond-dimension energy difference are
below `1e-10` per site.

## Problem

`mpskit_ground_state_j1j2` allocates a random `InfiniteMPS` on every call.
Independent random restarts make high-bond-dimension VUMPS relaxation very
slow and introduce initialization variation that is larger than the requested
energy tolerance.

## Design

### Continuation state

Extend `mpskit_ground_state_j1j2` to accept an optional initial
`InfiniteMPS`. When provided, it is optimized directly and the returned result
includes the final VUMPS residual. Existing calls keep the random-state
behavior unchanged.

### Bond-dimension sweep

The convergence script starts at the smallest requested `D`, then expands
the converged state by `changebonds(..., RandExpand(trscheme=truncrank(ΔD)))`
for each larger `D`. The enlarged state is refined with VUMPS; no stage
starts from a fresh random state after the first.

The script records after every completed stage:

- bond dimension and VUMPS residual;
- energy per site, correlation length, and entropy;
- the absolute energy change from the preceding bond dimension.

It writes the JSON checkpoint atomically after each stage so an interrupted
campaign preserves completed results.

### Acceptance criterion

For two consecutive bond dimensions `D_i` and `D_{i+1}`, accept convergence only
when both stages have residual below `state_tol = 1e-10` and

`abs(e(D_{i+1}) - e(D_i)) < 1e-10`.

The reported energy is the final, larger-`D` value. A VUMPS iteration cap is
reported as a failure to certify convergence rather than treated as success.

## Scope

- Modify the MPSKit extension only to support a continuation input and expose
  the terminal residual.
- Add a project workflow script and focused tests.
- Do not change the Hamiltonian, unit cell, model conventions, or default
  behavior of existing callers.

## Verification

1. Verify a same-`D` continuation returns the requested state and exposes a
   finite residual.
2. Verify the expansion step increases the virtual dimension by `ΔD`.
3. Run a small-`D` two-stage sweep and verify its JSON records both stages
   and their raw energy difference.
4. Run the `J2 = 0.5`, `Ly = 4` campaign and report its final residuals and
   last consecutive-(D) energy difference.
