# TFIM Energy Overlap Marker Design

## Purpose

Make closely overlapping IsoPEPS and DMRG points legible in the TFIM energy-versus-field figure without changing the calculated data, axes, legend placement, or line styles.

## Approved presentation

For reference series whose label begins with `DMRG`, retain the existing orange dotted line and square marker shape, but render the square as a 1.25× open marker: transparent fill with an orange outline. The IsoPEPS series remains a filled blue circle. When the values coincide, the blue circle is visible inside the orange square.

## Scope

Update only `plot_energy_error_vs_g` in `src/visualization/scan_comparisons.jl`, including the legend representation generated from its plotting handles. Apply the treatment to every DMRG reference plotted by that function so the style is consistent across its callers. Do not alter non-DMRG references, computed energies, scan inputs, or saved result data.

## Verification

Add a focused visualization test that creates a TFIM IsoPEPS/DMRG overlap, then verifies the DMRG scatter plot uses an open square with its own outline and a marker size larger than the filled IsoPEPS circle. Regenerate the requested PDF via the existing `energy-vs-g-tfim` target, render it to PNG, and visually confirm both marker types are distinguishable at the final publication size.
