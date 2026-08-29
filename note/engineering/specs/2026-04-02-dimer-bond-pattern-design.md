# Dimer-Dimer Correlation Bond Pattern Visualization

## Context

For detecting VBS (valence bond solid) order in the J1-J2 Heisenberg model, we need a real-space visualization of dimer-dimer correlations. The goal is to distinguish columnar VBS from plaquette VBS by showing the spatial pattern of connected dimer correlations relative to a reference bond on the cylinder lattice.

## Design

### Function Signature

```julia
function plot_dimer_bond_pattern(filename::String;
                                 max_cols::Int=10,
                                 ref_pos::Int=1,
                                 ref_col::Int=1,
                                 ref_orientation::Symbol=:vertical,
                                 save_path=nothing)
```

### Layout

A 2D grid representing the cylinder geometry:
- x-axis = column index (1 to `max_cols`)
- y-axis = row index (1 to `row`, periodic boundary)
- Sites drawn as small filled circles at grid points
- Bonds drawn as thick line segments between neighboring sites:
  - **Vertical bonds**: between `(pos, col)` and `(pos%row+1, col)` — same column, adjacent rows
  - **Horizontal bonds**: between `(pos, col)` and `(pos, col+1)` — same row, adjacent columns
- Bond color = connected dimer-dimer correlation `C_D(ref, bond) = ⟨D_ref D_bond⟩ - ⟨D_ref⟩⟨D_bond⟩`
- Reference bond highlighted with a thick black outline/marker
- Colormap: diverging `RdBu` (red = positive/ferro-dimer, blue = negative/anti-ferro-dimer, white = zero)
- Colorbar on the right

### Data Computation

1. Load result from JSON via `load_result(filename)`
2. Build `TransferOperator` (handle 1x1 and 2x2 unit cells)
3. Compute connected dimer-dimer correlations for all bonds relative to the reference:
   - **Same-orientation bonds**: Use `_dimer_cross_correlation(op, separations, orientation, ref_pos, target_pos)` for each target row position across column separations 0 to `max_cols-1`
   - **Cross-orientation bonds**: Currently not supported by the correlation functions. These bonds will be drawn in gray (no data) with a note in the legend.
4. Arrange correlation values into arrays indexed by `(pos, col, orientation)`

### Cylinder Periodicity

- Draw rows 1 to `row` with a dashed line connecting row `row` back to row 1 (indicating periodicity)
- Vertical bonds between row `row` and row 1 are drawn normally (they wrap around)

### Visual Details

- Site markers: small gray circles (radius ~3pt)
- Bond line width: proportional to `|C_D|` (min width 1pt, max width 5pt)
- Bond color: `RdBu` colormap, symmetric around zero
- Reference bond: drawn with black outline, linewidth=4
- Gray bonds: cross-orientation correlations (not computed)
- Title includes model parameters (J1, J2, row, p) and reference bond location
- `DataAspect()` for correct geometry

### Return Value

`(fig, correlation_data)` where `correlation_data` is a `Dict` with keys:
- `:vertical => Matrix{Float64}(row, max_cols)` — vertical bond correlations
- `:horizontal => Matrix{Float64}(row, max_cols-1)` — horizontal bond correlations

## Files to Modify

- `src/visualization.jl` — append `plot_dimer_bond_pattern` function
- `src/IsoPEPS.jl` — add to export line

## Verification

```julia
# Load a J1-J2 result in the intermediate regime
fig, data = plot_dimer_bond_pattern("project/results/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json";
    max_cols=10, ref_pos=1, ref_col=1, ref_orientation=:vertical)
display(fig)

# Check: VBS order shows alternating positive/negative correlations
# Columnar: staggered along columns
# Plaquette: 2x2 checkerboard pattern
```
