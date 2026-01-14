# Why is the Spectral Gap Small After Optimization?

## Problem Statement

After optimizing parameters for the ground state, all transfer matrix eigenvalues are close to 1, resulting in a very small spectral gap. This is problematic because:

- A small gap means slow convergence to the fixed point
- It suggests the transfer matrix may have multiple (near-)degenerate fixed points
- The correlation length ξ = 1/gap becomes very large

---

## Potential Causes

### 1. Symmetry-Induced Degeneracy

The ansatz in `gates.jl` uses a specific brick-wall CNOT structure:

```julia
function _build_layer(params, r, nqubits)
    # ... single qubit rotations ...
    
    # Nearest-neighbor CNOTs: (1,2), (2,3), (3,4), (4,5), ...
    cnot_nn = Matrix{ComplexF64}(I, dim, dim)
    for i in 1:nqubits-1
        cnot_nn *= Matrix(cnot(nqubits, i+1, i))
    end
    # ... more CNOTs ...
end
```

**Issue:** If the optimized parameters create gates with **Z₂ or other discrete symmetries**, the transfer matrix can have degenerate eigenvalues corresponding to different symmetry sectors. This is especially problematic if the target Hamiltonian (TFIM) has symmetry breaking in the ground state phase.

### 2. Energy Landscape Has Degenerate Minima

The energy function in `training.jl`:

```julia
energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
```

**Issue:** The optimizer minimizes energy **without any regularization on the spectral gap**. It may find a minimum where:
- Energy is locally optimal
- But the transfer matrix is nearly non-primitive (multiple dominant eigenvalues)

### 3. Optimization Converging to "Trivial" Solutions

If λ₂ ≈ λ₁ ≈ 1, the transfer matrix is close to having **multiple fixed points**. This can happen when:
- The gate U is close to a **block-diagonal** structure
- The tensor A decomposes into independent sectors

### 4. Parameter Space Issue

The Rx-Rz ansatz might not have enough expressiveness or has "flat" directions:

```julia
for i in 1:nqubits
    idx = 2*nqubits*r - 2*nqubits + 2*i - 1
    push!(single_qubit_gates, Yao.Rx(params[idx]) * Yao.Rz(params[idx+1]))
end
```

**Issue:** Rx-Rz only covers a subset of SU(2). Some rotations may be unreachable.

---

## Diagnostic Steps

### 1. Check Transfer Matrix Structure

```julia
function diagnose_transfer_matrix(gates, row, nqubits)
    T = get_transfer_matrix(gates, row, nqubits)
    
    # Check all eigenvalues
    eigenvalues = eigvals(T)
    sorted_mags = sort(abs.(eigenvalues), rev=true)
    
    println("Top 5 eigenvalue magnitudes:")
    for i in 1:min(5, length(sorted_mags))
        println("  λ_$i = $(sorted_mags[i])")
    end
    
    # Check if T is close to identity
    I_mat = Matrix(I, size(T)...)
    dist_to_identity = norm(T - I_mat) / norm(I_mat)
    println("Distance to identity: $dist_to_identity")
    
    # Check for block structure (off-diagonal magnitude)
    T_abs = abs.(T)
    off_diag_norm = norm(T_abs - Diagonal(diag(T_abs)))
    println("Off-diagonal norm: $off_diag_norm")
    
    # Check eigenvalue clustering
    n_close_to_1 = count(x -> abs(x - 1.0) < 0.1, sorted_mags)
    println("Eigenvalues within 0.1 of 1: $n_close_to_1")
    
    return sorted_mags
end
```

### 2. Check Gate Structure

```julia
function check_gate_structure(gate)
    U = Matrix(gate)
    println("Gate is unitary: ", U * U' ≈ I)
    
    # Check if close to identity
    println("Distance to I: ", norm(U - I))
    
    # Check singular values (should all be 1 for unitary)
    svs = svdvals(U)
    println("Singular values: ", svs)
    
    # Check eigenvalue distribution
    eigs = eigvals(U)
    println("Gate eigenvalues (phases): ", angle.(eigs))
end
```

---

## Suggested Fixes

### Option 1: Add Gap Regularization

Modify the objective to penalize small gaps:

```julia
function objective_with_gap(x; gap_weight=0.1)
    gates = build_unitary_gate(x, p, row, nqubits)
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    
    X_cost = real(compute_X_expectation(rho, gates, row, nqubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, nqubits)
    
    energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
    
    # Penalize small spectral gap (want λ₂ << 1)
    # gap = -log(λ₂), so larger gap is better
    gap_penalty = -gap_weight * gap
    
    return energy - gap_penalty  # Subtract because we're minimizing
end
```

### Option 2: Use Richer Ansatz

Replace Rx-Rz with full SU(2) coverage (Rz-Ry-Rz decomposition):

```julia
# Full SU(2): Rz-Ry-Rz decomposition (Euler angles)
push!(single_qubit_gates, 
      Yao.Rz(params[idx]) * Yao.Ry(params[idx+1]) * Yao.Rz(params[idx+2]))
```

This requires 3 parameters per qubit instead of 2, but covers all single-qubit unitaries.

### Option 3: Symmetry Breaking Initialization

Initialize parameters to explicitly break any symmetry:

```julia
# Add small random perturbations to break degeneracy
params_init = rand(n_params) .* 2π .+ 0.1 .* randn(n_params)
```

### Option 4: Multi-start Optimization

Run optimization from multiple random initializations and pick the one with the largest gap:

```julia
best_result = nothing
best_gap = 0.0

for trial in 1:10
    params_init = rand(n_params) .* 2π
    result = optimize_exact(params_init, J, g, p, row, nqubits)
    
    if result.gap > best_gap
        best_gap = result.gap
        best_result = result
    end
end
```

### Option 5: Check Physics at Different g Values

The TFIM has different phases:
- **g << J**: Ordered phase (Z-aligned), should have finite correlation length
- **g >> J**: Disordered phase (X-aligned), should have short correlation length  
- **g ≈ J**: Critical point, correlation length diverges

If you're near the critical point, a small gap is **expected physically**. Try optimizing at g=0.1 or g=3.0 first to verify the method works away from criticality.

---

## Quick Checklist

- [ ] Check eigenvalue spectrum: Are multiple eigenvalues ≈ 1?
- [ ] Check gate structure: Is the optimized gate close to identity or block-diagonal?
- [ ] Verify physics: Are you near the critical point (g ≈ J)?
- [ ] Try different initializations: Does the gap vary significantly?
- [ ] Add gap regularization: Does this improve results?
- [ ] Use richer ansatz: Does Rz-Ry-Rz give better gaps?
