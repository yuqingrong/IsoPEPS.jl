# IsoPEPS Optimization: Escaping Product State Local Minima

## Problem Description

During TFIM (Transverse Field Ising Model) optimization with IsoPEPS, the variational circuit often converges to **trivial fixed points** characterized by:

| Property | Trivial State | Expected Ground State |
|----------|---------------|----------------------|
| Physical state | Product state | Correlated/entangled |
| Virtual bonds | Maximally entangled | Moderate entanglement |
| ⟨X⟩ | ≈ 1 | < 1 (depends on g) |
| ⟨ZZ⟩_connected | ≈ 0 | < 0 (ferromagnetic) |
| Energy | ≈ -g | < -g |

### Why This Happens

1. **Stable Fixed Point**: Product states with maximally entangled virtual bonds are stable fixed points of the quantum channel
2. **Easy to Find**: Random initialization often lands in the basin of attraction of trivial states
3. **Local Minimum**: CMA-ES gets trapped because the energy landscape has many such minima
4. **Sampling Noise**: Stochastic gradients make it hard to distinguish between nearby local minima

### Mathematical Analysis

For a uniform MPS tensor `A[s, l, r]` with structure:
```
A[s, i, i] = f(s)  (diagonal in virtual indices)
```

The resulting state is:
```
|ψ⟩ ∝ ∑_{s₁...sₙ} ∏ᵢ f(sᵢ) |s₁...sₙ⟩ = ⊗ᵢ (∑ₛ f(s)|s⟩)
```

This is a **product state** despite having bond dimension D > 1. The entanglement entropy is:
- **Bond entanglement** (SVD of tensor): log(D) (maximal)
- **Physical entanglement** (bipartite cut): **0** (product state)

---

## Implemented Solutions

### 1. Smart Initialization (`initialize_tfim_params`)

New function to generate physics-informed initial parameters:

```julia
using IsoPEPS

# Mean-field initialization (recommended)
params = initialize_tfim_params(p, nqubits, g; mode=:meanfield)

# Entangled initialization (alternative)
params = initialize_tfim_params(p, nqubits, g; mode=:entangled)

# Random initialization (original behavior)
params = initialize_tfim_params(p, nqubits, g; mode=:random)
```

**Mean-field mode**: Uses angle `θ_mf = atan(1/g)` which approximates the mean-field ground state direction.

**Entangled mode**: Uses `π/4` rotations to create Bell-like entanglement from the start.

### 2. ZZ Correlation Regularization

New `zz_weight` parameter to penalize product states:

```julia
result = optimize_circuit(params, J, g, p, row, nqubits;
    zz_weight=0.5,  # Regularization strength
    # ... other parameters
)
```

**How it works**:
- Computes connected correlation: `⟨ZZ⟩_c = ⟨ZZ⟩ - ⟨Z⟩²`
- Product states have `⟨ZZ⟩_c ≈ 0`
- Ferromagnetic states have `⟨ZZ⟩_c < 0`
- Penalty: `max(0, ⟨ZZ⟩_c + 0.1)` (penalizes when correlations are too weak)

---

## Usage Example

```julia
using IsoPEPS

# Model parameters
J = 1.0
g = 2.5  # Below critical point g_c ≈ 3.04
row = 2
p = 4      # Circuit depth
nqubits = 5

# Smart initialization
params = initialize_tfim_params(p, nqubits, g; mode=:meanfield)

# Optimize with regularization
result = optimize_circuit(params, J, g, p, row, nqubits;
    sigma0=1.5,      # Larger step size for exploration
    popsize=30,      # Larger population
    zz_weight=0.3,   # Regularization
    maxiter=2000
)

# Check if converged to product state
Z = result.final_Z_samples
N = length(Z)
ZZ = mean(Z[i] * Z[i+1] for i in 1:N-1)
ZZ_connected = ZZ - mean(Z)^2

if abs(ZZ_connected) < 0.05
    @warn "Likely product state - consider rerunning with different initialization"
else
    @info "Non-trivial state found" ZZ_connected
end
```

---

## Additional Recommendations

### 1. Increase Circuit Depth

Shallow circuits (small `p`) cannot create sufficient entanglement:

| `p` | Expressivity | Recommendation |
|-----|-------------|----------------|
| 1-2 | Low | Only for testing |
| 3-4 | Medium | Standard optimization |
| 5-6 | High | Complex states near critical point |

### 2. Multi-Start Optimization

Run from multiple starting points:

```julia
best_result = nothing
best_energy = Inf

for trial in 1:10
    params = initialize_tfim_params(p, nqubits, g; mode=:meanfield)
    params .+= 0.3 * randn(length(params))
    
    result = optimize_circuit(params, J, g, p, row, nqubits;
        maxiter=500,
        zz_weight=0.3
    )
    
    if result.final_cost < best_energy
        best_energy = result.final_cost
        best_result = result
    end
end
```

### 3. CMA-ES Hyperparameters

| Parameter | Default | Recommended for escaping local minima |
|-----------|---------|--------------------------------------|
| `sigma0` | 1.0 | 1.5 - 2.0 (more exploration) |
| `popsize` | auto | 30-50 (better coverage) |
| `maxiter` | 5000 | Start with 500-1000 per trial |

### 4. Diagnostic Checks

After optimization, verify the state is non-trivial:

```julia
function diagnose_state(result)
    Z = result.final_Z_samples
    X = result.final_X_samples
    
    Z_mean = mean(Z)
    X_mean = mean(X)
    ZZ_mean = mean(Z[i] * Z[i+1] for i in 1:length(Z)-1)
    ZZ_connected = ZZ_mean - Z_mean^2
    
    println("=== State Diagnostics ===")
    println("⟨Z⟩ = $(round(Z_mean, digits=4))")
    println("⟨X⟩ = $(round(X_mean, digits=4))")
    println("⟨ZZ⟩ = $(round(ZZ_mean, digits=4))")
    println("⟨ZZ⟩_c = $(round(ZZ_connected, digits=4))")
    println()
    
    if abs(ZZ_connected) < 0.05
        println("⚠️  WARNING: Likely product state")
        return false
    else
        println("✓ Non-trivial correlated state")
        return true
    end
end
```

---

## Physical Context

### TFIM Phase Diagram (2D)

```
        Paramagnetic (PM)          Ferromagnetic (FM)
    ⟨X⟩ ≈ 1, ⟨Z⟩ ≈ 0           ⟨X⟩ < 1, ⟨Z⟩ ≠ 0
    
    |--------------------------|--------------------------|
    g = 0                    g_c ≈ 3.04                 g → ∞
                        (critical point)
```

- **g > g_c**: Ground state is paramagnetic, closer to product state
- **g < g_c**: Ground state has ferromagnetic order, significant correlations
- **g ≈ g_c**: Critical point, long-range correlations, hardest to optimize

### Expected Behavior

| g/J | Phase | ⟨ZZ⟩_connected | Difficulty |
|-----|-------|----------------|------------|
| > 4 | PM | Small negative | Easy |
| 2-4 | Near critical | Moderate negative | Hard |
| < 2 | FM | Large negative | Medium |

---

## Summary

The product state problem occurs because:
1. Trivial states are stable fixed points
2. Random initialization often finds them
3. CMA-ES struggles to escape shallow local minima

Solutions implemented:
1. **`initialize_tfim_params`**: Physics-informed initialization
2. **`zz_weight`**: Regularization to penalize product states

Best practices:
- Use mean-field initialization
- Enable ZZ regularization (zz_weight=0.3-0.5)
- Run multiple trials with different random seeds
- Increase population size for better exploration
- Check `⟨ZZ⟩_connected` after optimization

---

*Report generated: 2026-01-25*
*IsoPEPS.jl variational optimization troubleshooting*

