# visualize

Generate plots and visualizations for quantum states and optimization results.

## Usage
```
/visualize <plot_type> [options]
```

## Plot Types
- `state`: Visualize quantum state structure
- `convergence`: Plot optimization convergence
- `energy`: Energy landscape or evolution
- `correlation`: Correlation functions
- `entanglement`: Entanglement structure
- `tensor-network`: Tensor network diagram

## Options
- `--save <path>`: Save figure to file
- `--format <fmt>`: Output format (png, pdf, svg)
- `--style <name>`: Plot style preset
- `--interactive`: Create interactive plot

## Workflow

1. **Load data**
   - Read state or optimization results
   - Parse data format
   - Validate data integrity

2. **Prepare visualization**
   - Choose appropriate plot type
   - Set up figure and axes
   - Configure color schemes and styles

3. **Generate plot**
   - Use CairoMakie for rendering
   - Apply styling
   - Add labels, legends, annotations

4. **Save or display**
   - Save to file if requested
   - Display in notebook/REPL if interactive
   - Generate multiple formats if needed

## Plot Type Details

### State Visualization
- Tensor network structure
- Bond dimensions
- Entanglement pattern
- Schmidt values

### Convergence Plot
- Objective function vs iteration
- Parameter evolution
- Gradient norms
- Constraint violations

### Energy Plot
- Energy vs parameter
- Energy landscape (2D/3D)
- Energy evolution during optimization
- Comparison with exact results

### Correlation Functions
- Two-point correlations
- Correlation length
- Spatial patterns
- Time evolution

### Entanglement
- Entanglement entropy
- Mutual information
- Entanglement spectrum
- Area law verification

### Tensor Network Diagram
- Node and edge structure
- Bond dimensions labeled
- Contraction order visualization
- Computational complexity

## Example Usage

```julia
# Plot convergence
/visualize convergence --save results/convergence.pdf

# Interactive energy landscape
/visualize energy --interactive --style publication

# Correlation function
/visualize correlation --save correlations.png --format png

# Tensor network structure
/visualize tensor-network --save network.svg --format svg
```

## Styling Presets
- `default`: Standard Makie theme
- `publication`: High-quality for papers (vector graphics)
- `presentation`: Large fonts for slides
- `dark`: Dark background theme
- `minimal`: Minimal styling

## Example Code

```julia
using CairoMakie

function plot_convergence(history)
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel="Iteration",
        ylabel="Energy",
        title="Optimization Convergence")

    lines!(ax, 1:length(history), history,
        color=:blue, linewidth=2)

    save("convergence.pdf", fig)
    return fig
end
```

## Success Criteria
- Plot accurately represents data
- Styling is appropriate for use case
- Labels and legends are clear
- File is saved in requested format
- Resolution is suitable for intended use
