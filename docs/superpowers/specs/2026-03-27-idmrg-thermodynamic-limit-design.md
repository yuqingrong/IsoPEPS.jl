# iDMRG Thermodynamic Limit Reference for Quantum Channel Comparison

## Problem

The quantum channel in IsoPEPS represents an infinite Ly=4 cylinder. We need the exact ground state energy of this system as a reference for evaluating the quantum channel's variational accuracy.

Key insight: the MPS bond dimension D in iDMRG is NOT the same as the PEPS bond dimension D in the quantum channel. The MPS D must be large enough for the iDMRG energy to converge — this gives the true ground state of the Ly=4 cylinder, which is the ground truth the quantum channel (with PEPS D=2) is trying to approximate.

## Solution

Use the existing `mpskit_ground_state_j1j2` function (VUMPS on InfiniteCylinder) with large MPS bond dimension (D=64, increase until convergence) to get the exact ground state energy of the Ly=4 cylinder. This serves as ground truth for quantum channel comparison.

## Existing Code

- `ext/MPSKitExt.jl:83-127` — `mpskit_ground_state_j1j2(d, D, J1, J2, row)`: VUMPS on InfiniteCylinder, returns energy/site, correlation length, entropy, spectrum
- `project/mpskit_j1j2.jl` — script that runs J2 scan with D=16, row=4

## Changes

### 1. Update `project/mpskit_j1j2.jl` with converged reference scan

- Run with large MPS D (64, 128, 256) until energy per site converges (delta < 1e-6)
- Save results to JSON for comparison with quantum channel
- Scan J2 values matching the quantum channel parameter scan

JSON output format:
```json
{
  "parameters": {"model": "heisenberg_j1j2", "Ly": 4, "method": "iDMRG_VUMPS"},
  "scan_param": "J2",
  "scan_values": [0.0, 0.1, ...],
  "D_values": [64, 128, 256],
  "energies_per_site": {"64": [-0.50, ...], "128": [-0.50, ...], "256": [-0.50, ...]},
  "converged_energies_per_site": [-0.50, ...],
  "correlation_lengths": [1.2, ...],
  "entropies": [0.5, ...]
}
```

### 2. Add iDMRG reference overlay to postprocess plots

In `project/postprocess.jl`, add utility to overlay converged iDMRG reference energy on quantum channel training plots. This shows how close the quantum channel (PEPS D=2) energy is to the exact Ly=4 cylinder ground state.

### 3. No changes to MPSKitExt

The existing `mpskit_ground_state_j1j2` function already does everything needed.

## Key Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| d | 2 | Spin-1/2 |
| MPS D | 64 → 256 (converge) | Large enough to get exact Ly=4 cylinder ground state |
| row | 4 | Cylinder circumference = Ly |
| unit_cell_cols | 2 | 2x2 unit cell matching quantum channel |
| J1 | 1.0 | Fixed |
| J2 | scan | Match quantum channel scan values |

## Clarification: MPS D vs PEPS D

- **MPS D (iDMRG)**: Controls accuracy of the 1D MPS approximation of the 2D cylinder ground state. Must be large (D ~ 64-256) for convergence on Ly=4.
- **PEPS D (quantum channel)**: The tensor network bond dimension of the IsoPEPS ansatz. D=2 is the ansatz being tested.
- The iDMRG reference with converged MPS D gives the exact target energy that the PEPS D=2 ansatz is trying to reach.

## Clarification: samples vs lattice size

The quantum channel's `samples=10000` is the number of measurement shots for statistical averaging, NOT the lattice size. The quantum channel represents the same infinite cylinder regardless of sample count.

## Verification

1. Run `mpskit_ground_state_j1j2(2, 64, 1.0, 0.5, 4)` and verify convergence
2. Increase D to 128, 256 and confirm energy converges (delta < 1e-6)
3. Compare converged energy with quantum channel results
4. Overlay iDMRG reference on quantum channel training plots
