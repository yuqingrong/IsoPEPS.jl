# Gradient Computation for Isometric PEPS Circuits

## Problem Setup

In the IsoPEPS circuit optimization, parameterized gates are built using `Rx(θ)` and `Rz(θ)` rotations:

```julia
# From gates.jl
push!(single_qubit_gates, Yao.Rx(params[idx]) * Yao.Rz(params[idx+1]))
```

When `share_params=true`, the same parameters are shared across all `row` gates, and each gate is applied repeatedly in the transfer matrix to reach the thermodynamic limit (~1000 applications).

---

## Classical Simulation vs. Quantum Execution

### Important Distinction

| Scenario | What Happens | Quantum Advantage? |
|----------|--------------|-------------------|
| **Classical simulation** (current code) | Construct gate matrices, contract tensors | ❌ No |
| **Quantum hardware** | Execute circuit, measure outcomes | ✅ Yes (for large systems) |

### Classical Simulation (Current Code)

Everything is done classically:
- `build_unitary_gate` → constructs matrices
- `compute_transfer_spectrum` → contracts tensors
- Gradients → computed through classical operations

**For classical simulation**: Direct tensor network methods (VUMPS, adjoint differentiation) are more natural.

### On Quantum Hardware

The circuit is **executed, not constructed**:

```
Classical Computer                  Quantum Hardware
─────────────────                  ────────────────
θ (parameters)        ──────►      Run circuit U(θ)
                                         │
                                         ▼
                                   Measure outcomes
                                         │
                     ◄──────       Samples {+1, -1}
Estimate ⟨E⟩ from samples
```

The quantum device **naturally prepares** |ψ(θ)⟩ = U(θ)|0⟩ without exponential classical cost.

### Why Use Circuit Structure for Classical Optimization?

1. **Future quantum deployment** — hardware-efficient ansatz
2. **Natural parameterization** — rotation angles have clear meaning
3. **Benchmarking** — compare classical vs quantum performance

---

## Method 1: Parameter-Shift Rule (Naive)

### Formula

For gates $U(\theta) = e^{-i\theta P/2}$ (Pauli rotation gates):

$$\frac{\partial \langle E \rangle}{\partial \theta_j} = \frac{1}{2}\left[\langle E \rangle_{\theta_j + \pi/2} - \langle E \rangle_{\theta_j - \pi/2}\right]$$

### Key Property: Works with Shared Parameters

When a parameter $\theta$ appears at $k$ locations, shifting globally gives the **correct total gradient** (sum of all partial derivatives) automatically.

### Code

```julia
function gradient_parameter_shift(params, J, g, p, row, nqubits; share_params=true)
    n_params = length(params)
    grad = zeros(n_params)
    
    for j in 1:n_params
        params_plus = copy(params); params_plus[j] += π/2
        params_minus = copy(params); params_minus[j] -= π/2
        
        E_plus = compute_energy_from_params(params_plus, J, g, p, row, nqubits; share_params)
        E_minus = compute_energy_from_params(params_minus, J, g, p, row, nqubits; share_params)
        
        grad[j] = (E_plus - E_minus) / 2
    end
    return grad
end
```

### Limitation

❌ **Expensive for thermodynamic limit**: If parameters appear in ~1000 transfer matrix applications, naive evaluation requires full contraction each time.

---

## Method 2: Adjoint Differentiation Through Fixed Point (Efficient)

### Key Insight

The transfer matrix approach computes the **fixed point** $\rho$ satisfying:

$$T(\theta)|\rho\rangle = \lambda|\rho\rangle$$

This fixed point represents the infinite chain limit. We can differentiate **through the eigenvalue equation** instead of through 1000 explicit applications.

### Mathematical Formulation

For the fixed point equation, the derivative is:

$$\frac{\partial \rho}{\partial \theta} = (\lambda I - T)^{+} \frac{\partial T}{\partial \theta} |\rho\rangle$$

where $(\lambda I - T)^{+}$ is the pseudo-inverse (with $\rho$ projected out).

### Full Gradient via Chain Rule

$$\frac{\partial E}{\partial \theta} = \underbrace{\frac{\partial E}{\partial \rho} \cdot \frac{\partial \rho}{\partial \theta}}_{\text{through fixed point}} + \underbrace{\frac{\partial E}{\partial T} \cdot \frac{\partial T}{\partial \theta}}_{\text{direct gate terms}}$$

### Code Sketch

```julia
function gradient_adjoint(params, J, g, p, row, nqubits; ε=1e-7)
    n_params = length(params)
    grad = zeros(n_params)
    
    # Forward pass
    gates = build_unitary_gate(params, p, row, nqubits; share_params=true)
    rho, λ_dom, _ = compute_transfer_spectrum(gates, row, nqubits)
    
    # Compute ∂E/∂ρ (adjoint seed)
    E0, ∂E_∂rho = compute_energy_and_adjoint(rho, gates, row, nqubits, g, J)
    
    for j in 1:n_params
        # ∂T/∂θⱼ via finite difference
        params_ε = copy(params); params_ε[j] += ε
        gates_ε = build_unitary_gate(params_ε, p, row, nqubits; share_params=true)
        
        # Solve (λI - T)·∂ρ = ∂T·ρ
        ∂rho = solve_fixed_point_derivative(gates, gates_ε, rho, λ_dom, ε)
        
        # Chain rule
        grad[j] = dot(∂E_∂rho, ∂rho) + direct_terms(...)
    end
    return grad
end

function solve_fixed_point_derivative(gates, gates_ε, rho, λ, ε)
    # Compute ∂T·ρ
    T_rho = apply_transfer_matrix(gates, rho)
    T_ε_rho = apply_transfer_matrix(gates_ε, rho)
    ∂T_rho = (T_ε_rho - T_rho) / ε
    
    # Solve shifted linear system
    shifted_T(v) = λ * v - apply_transfer_matrix(gates, v)
    ∂rho, _ = linsolve(shifted_T, ∂T_rho)
    return ∂rho
end
```

---

## Method 3: Automatic Differentiation (Zygote + KrylovKit)

### Easiest Approach

KrylovKit.jl (used in `compute_transfer_spectrum`) has built-in AD rules via ChainRules.jl:

```julia
using Zygote

function energy_from_params(params, J, g, p, row, nqubits)
    gates = build_unitary_gate(params, p, row, nqubits; share_params=true)
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    X_cost = real(compute_X_expectation(rho, gates, row, nqubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, nqubits)
    return real(-g * X_cost - J * (row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz))
end

# Automatic gradient!
grad = Zygote.gradient(p -> energy_from_params(p, J, g, p_depth, row, nqubits), params)[1]
```

### Requirements

- All operations must be AD-compatible
- May need custom `rrule` for `OMEinsum` contractions
- KrylovKit's `eigsolve` is already differentiable

---

## Method 4: Finite Differences (Baseline)

Simple but noisy:

```julia
function gradient_finite_diff(params, J, g, p, row, nqubits; ε=1e-5)
    E0 = compute_energy_from_params(params, J, g, p, row, nqubits)
    grad = zeros(length(params))
    
    for j in 1:length(params)
        params_ε = copy(params); params_ε[j] += ε
        grad[j] = (compute_energy_from_params(params_ε, J, g, p, row, nqubits) - E0) / ε
    end
    return grad
end
```

---

## Comparison Summary

| Method | Cost | Accuracy | Implementation |
|--------|------|----------|----------------|
| **Parameter-shift (naive)** | 2 × n_params × full contraction | Exact | Easy |
| **Adjoint through fixed point** | n_params × (1 forward + 1 linear solve) | Exact | Medium |
| **Zygote AD** | ~1 backward pass | Exact | Easy (if compatible) |
| **Finite difference** | n_params × full contraction | Noisy (O(ε)) | Trivial |
| **Parameter-shift + truncation** | 2 × n_params × L (L << 1000) | High | Easy |
| **SPSA** | 2 evaluations total | Noisy | Very easy |

---

## Recommendation

### For Classical Simulation (Current Code)

1. **First try**: Zygote AD with KrylovKit (easiest if it works)
2. **If AD fails**: Implement adjoint differentiation through the fixed point
3. **Fallback**: Parameter-shift with truncated depth + caching

The adjoint method is the standard approach in tensor network libraries (DMRG, VUMPS, PEPSKit) for differentiating through fixed-point iterations.

### For Quantum Hardware

1. **Use parameter-shift rule** — hardware-native, no classical simulation needed
2. **Apply truncation** — only need ~5ξ circuit depth for good gradient estimate
3. **Consider SPSA** — if parameter count is large and circuit execution is costly

---

## Cost Reduction for Parameter-Shift with Repeated Gates

When gates repeat ~1000 times (thermodynamic limit), naive parameter-shift is expensive. Here are techniques to reduce cost:

### Key Physics Insight: Correlation Length

The transfer matrix converges exponentially fast:

$$\|T^n \rho_0 - \rho_{\infty}\| \sim e^{-n/\xi}$$

where correlation length $\xi \sim 1/\text{gap}$.

**If gap ~ 0.1, then ξ ~ 10, NOT 1000!**

---

### Cost Reduction Method 1: Truncated Depth

Only contract $L \sim O(\xi)$ layers instead of 1000:

```julia
function gradient_truncated_parameter_shift(params, J, g, p, row, nqubits; 
                                             truncation_depth=50)  # Not 1000!
    grad = zeros(length(params))
    
    for j in 1:length(params)
        params_plus = copy(params); params_plus[j] += π/2
        params_minus = copy(params); params_minus[j] -= π/2
        
        # Use finite depth instead of full convergence
        E_plus = compute_energy_truncated(params_plus, truncation_depth, ...)
        E_minus = compute_energy_truncated(params_minus, truncation_depth, ...)
        
        grad[j] = (E_plus - E_minus) / 2
    end
    return grad
end

function compute_energy_truncated(params, L, J, g, p, row, nqubits)
    gates = build_unitary_gate(params, p, row, nqubits; share_params=true)
    
    # Start from some initial state
    rho = initialize_boundary_state(nqubits)
    
    # Only apply transfer matrix L times (not until convergence)
    for _ in 1:L
        rho = apply_transfer_matrix(gates, rho, row, nqubits)
        rho ./= norm(rho)
    end
    
    return compute_energy_from_rho(rho, gates, g, J, row, nqubits)
end
```

**Error bound**: $O(e^{-L \cdot \text{gap}})$ — exponentially small!

**Adaptive truncation**:
```julia
L = ceil(Int, 5 / gap)  # ~5 correlation lengths is enough
```

---

### Cost Reduction Method 2: SPSA (Simultaneous Perturbation)

Only **2 circuit evaluations** regardless of parameter count:

```julia
function gradient_spsa(params, J, g, p, row, nqubits; c=0.1)
    n = length(params)
    
    # Random perturbation direction (Rademacher: ±1)
    Δ = rand([-1, 1], n)
    
    # Only 2 evaluations total!
    E_plus = compute_energy_from_params(params .+ c .* Δ, J, g, p, row, nqubits)
    E_minus = compute_energy_from_params(params .- c .* Δ, J, g, p, row, nqubits)
    
    # Stochastic gradient estimate
    grad = (E_plus - E_minus) / (2c) .* Δ
    return grad
end
```

**Cost**: 2 evaluations vs. 2n for parameter-shift!

---

### Cost Reduction Method 3: Stochastic Coordinate Descent

Update random subset of parameters each iteration:

```julia
function gradient_stochastic_coordinate(params, J, g, p, row, nqubits; 
                                         batch_size=5)
    n = length(params)
    grad = zeros(n)
    
    # Randomly select parameters to update
    indices = randperm(n)[1:batch_size]
    
    for j in indices
        params_plus = copy(params); params_plus[j] += π/2
        params_minus = copy(params); params_minus[j] -= π/2
        
        E_plus = compute_energy_from_params(params_plus, J, g, p, row, nqubits)
        E_minus = compute_energy_from_params(params_minus, J, g, p, row, nqubits)
        
        grad[j] = (E_plus - E_minus) / 2
    end
    return grad
end
```

---

### Cost Reduction Method 4: Caching Transfer Matrix

When shifting one parameter, most computation is unchanged:

```julia
function gradient_with_caching(params, J, g, p, row, nqubits)
    # Pre-compute base quantities
    gates_base = build_unitary_gate(params, p, row, nqubits; share_params=true)
    rho_base, λ_base, _ = compute_transfer_spectrum(gates_base, row, nqubits)
    
    grad = zeros(length(params))
    
    for j in 1:length(params)
        # Only rebuild affected layer(s)
        layer_idx = get_layer_index(j, nqubits)
        
        # Warm-start from base fixed point (faster convergence)
        T_plus = build_transfer_shifted(params, j, +π/2, ...)
        T_minus = build_transfer_shifted(params, j, -π/2, ...)
        
        rho_plus = refine_fixed_point(T_plus, rho_base)  # Few iterations
        rho_minus = refine_fixed_point(T_minus, rho_base)
        
        E_plus = compute_energy_from_rho(rho_plus, ...)
        E_minus = compute_energy_from_rho(rho_minus, ...)
        
        grad[j] = (E_plus - E_minus) / 2
    end
    return grad
end
```

---

### Cost Reduction Method 5: Perturbative Gradient

Since the gate repeats identically, perturbation at position $i$ decays as $e^{-|i|/\xi}$:

$$\frac{\partial E}{\partial \theta} = \sum_{i} e^{-|i|/\xi} \cdot (\text{local gradient at } i) \approx \frac{1}{1 - e^{-1/\xi}} \cdot (\text{single-site gradient})$$

```julia
function gradient_perturbative(params, J, g, p, row, nqubits; gap)
    ξ = 1 / gap  # Correlation length
    amplification = 1 / (1 - exp(-1/ξ))
    
    # Compute gradient as if gate appears only ONCE
    grad_single = gradient_single_site(params, J, g, p, row, nqubits)
    
    # Scale by geometric series factor
    return amplification * grad_single
end
```

---

### Cost Reduction Comparison

| Method | Cost | Accuracy | Best When |
|--------|------|----------|-----------|
| **Truncated depth** | 2n × L (L << 1000) | High if gap > 0 | Gap is known |
| **SPSA** | 2 total | Noisy estimate | Many parameters |
| **Stochastic coordinate** | 2 × batch_size | Exact for subset | Iterative optimization |
| **Caching** | Reduced per-param | Exact | Reusable structure |
| **Perturbative** | 2n × 1 | Approx (gap-dependent) | Large gap |

---

### Recommended Approach

**Combine truncated depth + caching**:

```julia
# Adaptive truncation based on spectral gap
function gradient_efficient(params, J, g, p, row, nqubits; gap=0.1)
    L = ceil(Int, 5 / gap)  # ~5 correlation lengths
    
    # Pre-compute base state for warm-starting
    gates_base = build_unitary_gate(params, p, row, nqubits)
    rho_base = compute_truncated_fixed_point(gates_base, L)
    
    grad = zeros(length(params))
    for j in 1:length(params)
        # Warm-start from base, only need few refinement steps
        E_plus = compute_energy_warm_start(params, j, +π/2, rho_base, L÷2)
        E_minus = compute_energy_warm_start(params, j, -π/2, rho_base, L÷2)
        grad[j] = (E_plus - E_minus) / 2
    end
    return grad
end
```

---

## References

- Mitarai et al. (2018) - Quantum circuit learning, parameter-shift rule
- Luchnikov et al. - Differentiable programming for quantum computing  
- Automatic differentiation of dominant eigensolver (KrylovKit.jl)
- Spall (1992) - SPSA: Multivariate stochastic approximation
- Hauru et al. - Riemannian optimization of tensor networks
