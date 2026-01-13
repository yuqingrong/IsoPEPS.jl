# Action Plan for Fixing Small Spectral Gaps

## Immediate Actions (Do These First)

### 1. Increase Circuit Depth for row=2

The most direct fix is to give your ansatz more expressibility:

```julia
# In your training script, change:
p = 4  # Current
# To:
p = 6  # or even 8

# Run optimization again:
result = optimize_circuit_evolutionary(
    g=0.5, J=1.0, row=2, nqubits=3, p=6,  # <-- Changed p
    maxiter=2000,  # Also increase iterations
    popsize=40,    # Larger population
    ...
)
```

**Why this helps**: More parameters → better chance of finding a solution with good spectral structure.

### 2. Use Hierarchical Initialization

Train row=1 first, then use it to initialize row=2:

```julia
# Step 1: Optimize row=1
println("Training row=1...")
result_row1 = optimize_circuit_evolutionary(
    g=0.5, J=1.0, row=1, nqubits=3, p=4,
    maxiter=1000, ...
)

# Verify good gap
gates1 = build_unitary_gate(result_row1.final_params, 4, 1, 3)
_, gap1, _ = compute_transfer_spectrum(gates1, 1, 3)
println("Row=1 gap: $gap1")  # Should be > 0.1

# Step 2: Use row=1 params as starting point for row=2
# (Duplicate parameters for each row)
initial_params_row2 = vcat(result_row1.final_params, result_row1.final_params)

println("Training row=2 with warm start...")
result_row2 = optimize_circuit_evolutionary(
    g=0.5, J=1.0, row=2, nqubits=3, p=4,
    initial_params=initial_params_row2,  # <-- Warm start
    maxiter=2000,  # More iterations
    ...
)
```

### 3. Multi-Start with Gap Filtering

Run multiple optimizations and keep the one with the best gap:

```julia
using Random

best_result = nothing
best_gap = 0.0
best_energy = Inf

for trial in 1:5
    Random.seed!(1000 + trial)
    
    println("\n=== Trial $trial/5 ===")
    result = optimize_circuit_evolutionary(
        g=0.5, J=1.0, row=2, nqubits=3, p=4,
        maxiter=1000, ...
    )
    
    # Compute spectral gap
    gates = build_unitary_gate(result.final_params, 4, 2, 3)
    _, gap, _ = compute_transfer_spectrum(gates, 2, 3)
    
    println("  Energy: $(result.final_cost)")
    println("  Gap: $gap")
    println("  ξ: $(round(1/gap, digits=1))")
    
    # Select based on gap AND energy
    if gap > best_gap && result.final_cost < best_energy + 0.1
        best_gap = gap
        best_result = result
        best_energy = result.final_cost
        println("  ★ New best!")
    end
end

println("\nBest result: energy=$best_energy, gap=$best_gap")
```

## Medium-Term Improvements

### 4. Add Gap Regularization to Loss Function

Modify your loss function to explicitly encourage large gaps:

**Create new file**: `src/regularized_training.jl`

```julia
"""
    compute_energy_with_gap_penalty(params, g, J, p, row, nqubits; λ_gap=0.01)

Compute energy with spectral gap regularization.
"""
function compute_energy_with_gap_penalty(params, g, J, p, row, nqubits; 
                                          λ_gap=0.01, share_params=true)
    # Build gates
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    
    # Sample and compute energy (existing code)
    rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                        conv_step=100, 
                                                        samples=1000)
    energy = compute_energy(X_samples, Z_samples, g, J, row)
    
    # Compute spectral gap (using fast iterative method)
    _, gap, _ = compute_transfer_spectrum(gates, row, nqubits; 
                                           num_eigenvalues=2,
                                           use_iterative=:always)
    
    # Penalty: want to maximize gap = minimize -gap
    gap_penalty = -gap
    
    # Combined loss
    loss = energy + λ_gap * gap_penalty
    
    return loss
end
```

Then use this in your optimization:

```julia
# Define objective for optimizer
function objective(params)
    return compute_energy_with_gap_penalty(
        params, g, J, p, row, nqubits;
        λ_gap=0.05  # Tune this hyperparameter
    )
end

# Run optimization with this modified objective
result = my_optimizer(objective, initial_params, ...)
```

### 5. Check Different Parameter Regimes

Some parameter regimes are inherently easier:

```julia
# Test which parameter ranges work well
test_params = [
    (g=0.5, row=1),   # Known good
    (g=0.5, row=2),   # Known bad
    (g=1.0, row=1),   # Test
    (g=1.0, row=2),   # Test
    (g=2.0, row=1),   # Test (may be harder)
]

for (g, row) in test_params
    result = optimize_and_check_gap(g, row, ...)
    # Log results for analysis
end
```

## Long-Term Solutions

### 6. Architecture Changes

Consider different circuit architectures:

- **Non-shared parameters**: Each row has its own parameters
  ```julia
  gates = build_unitary_gate(params, p, row, nqubits; share_params=false)
  ```
  This gives more flexibility but requires more parameters.

- **Different gate types**: Try other parametrizations beyond the current one

### 7. Physics-Informed Constraints

Use domain knowledge to guide optimization:

```julia
# Example: Initialize with product state
function initialize_product_state(p, row, nqubits)
    # Parameters that give approximately |+⟩⊗|+⟩⊗... state
    params = ...
    return params
end
```

## Diagnostic and Monitoring

### Use the Diagnostic Script Regularly

After each training run:

```julia
include("project/diagnose_spectral_gap.jl")

# Diagnose your new result
diagnose_spectral_gap("project/results/your_new_result.json")
```

### Add Gap Logging to Training

Modify your training loop to log gaps periodically:

```julia
# In training loop
if iteration % 50 == 0
    gates = build_unitary_gate(current_params, p, row, nqubits)
    _, gap, _ = compute_transfer_spectrum(gates, row, nqubits)
    println("Iteration $iteration: energy=$energy, gap=$gap")
end
```

## Expected Results

With these changes, you should see:

| Method | Expected gap (row=2, g=0.5) | Correlation length ξ |
|--------|----------------------------|---------------------|
| Current (p=4) | 0.0001 | ~9000 ❌ |
| Increased p=6 | 0.001-0.01 | 100-1000 ⚠️ |
| With warm start | 0.01-0.05 | 20-100 ⚠️ |
| With regularization | 0.05-0.2 | 5-20 ✓ |

Even achieving gap ~ 0.01 (ξ ~ 100) would be a **100x improvement**!

## Quick Test

Try this quick experiment right now:

```julia
using IsoPEPS
include("project/diagnose_spectral_gap.jl")

# Compare p=4 vs p=6 for row=2, g=0.5
for p in [4, 6]
    println("\n=== Testing p=$p ===")
    result = optimize_circuit_evolutionary(
        g=0.5, J=1.0, row=2, nqubits=3, p=p,
        maxiter=1000, popsize=30,
        samples_per_run=1000, n_parallel_runs=10
    )
    
    # Check gap
    gates = build_unitary_gate(result.final_params, p, 2, 3)
    _, gap, _ = compute_transfer_spectrum(gates, 2, 3)
    
    println("Energy: $(result.final_cost)")
    println("Gap: $gap")
    println("ξ: $(round(1/gap, digits=1))")
end
```

## Questions to Explore

1. **Does your loss landscape have many local minima with poor gaps?**
   - Test: Run multiple random initializations

2. **Is gap ~ 0.0001 physically correct for this system?**
   - Test: Compare with exact diagonalization or other methods

3. **Can we predict which (g, row) combinations will have gap issues?**
   - Test: Systematic scan across parameter space

## Summary

**Priority 1 (Do Now)**:
- ✅ Use diagnostic script to identify problematic cases  
- ⭐ Try p=6 or p=8 for row=2
- ⭐ Use hierarchical initialization (row=1 → row=2)
- ⭐ Multi-start with gap selection

**Priority 2 (This Week)**:
- Add gap regularization to loss
- Systematic testing across parameter space
- Log gaps during training

**Priority 3 (Future)**:
- Architecture explorations
- Physics-informed initialization
- Better optimization algorithms

The good news: This is a solvable problem! The diagnostic shows that row=1 works great (gap=0.43), so we know the method works. We just need to help the optimizer find better solutions for row=2.

