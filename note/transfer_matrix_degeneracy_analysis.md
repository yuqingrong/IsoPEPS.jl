# Transfer Matrix Degeneracy Analysis

## Summary

**Date**: January 2026  
**Model**: TFIM (Transverse Field Ising Model) on IsoPEPS  
**Parameters**: J=1.0, row=3, nqubits=3, p=3, D=2

## Key Observation

Regardless of the `zz_weight` regularization setting (0 or 5), the **virtual transfer matrix spectrum shows near-degeneracy** with all eigenvalues close to 1:

```
Virtual transfer matrix eigenvalues: [0.995, 0.995, 0.995, 0.995]
```

Meanwhile, the **physical channel** exhibits a **large spectral gap**:

| Metric | Value |
|--------|-------|
| Virtual channel gap | ~0.001 |
| Physical channel gap | ~7.5 |
| Virtual bond entropy | ~2.0 nats |
| Physical entropy | ~0.7 nats |
| S_max = log(2^4) | 2.77 nats |

---

## Physical Interpretation

### 1. Virtual Space Degeneracy

The near-degenerate virtual transfer matrix spectrum (all |λ| ≈ 1) indicates:

- **No preferred direction** in the virtual boundary space
- The MPS representation has **gauge redundancy** — many equivalent representations exist
- Virtual correlations do not decay (correlation length → ∞ in virtual space)

### 2. Physical Channel Gap

The large physical channel gap (~7.5) indicates:

- Physical correlations **decay rapidly**
- The physical state is **close to a product state**
- The fixed point of the physical channel is nearly pure

### 3. The Paradox: High Virtual Entanglement, Low Physical Entanglement

This combination reveals a fundamental property of isometric PEPS:

```
┌─────────────────────────────────────────────────────────────┐
│  Virtual Space          │  Physical Space                   │
├─────────────────────────┼───────────────────────────────────┤
│  Highly entangled       │  Near product state               │
│  (S_virtual ≈ 2.0)      │  (S_physical ≈ 0.7)               │
│  Degenerate spectrum    │  Large gap                        │
│  Many gauge choices     │  Unique fixed point               │
└─────────────────────────────────────────────────────────────┘
```

The isometric gates can create large virtual entanglement while the **physical state remains simple**. This is because the isometry condition allows "hiding" entanglement in the gauge degrees of freedom.

---

## Why `zz_weight` Regularization Doesn't Help

The `zz_weight` parameter in the cost function:

```julia
cost = energy + zz_weight * max(0.0, 0.1 - abs(ZZ_connected))
```

is designed to penalize weak ZZ correlations (product-state-like solutions). However:

### Problem 1: Wrong Target

- `zz_weight` penalizes weak **physical** ZZ correlations
- The degeneracy is in the **virtual** space
- Physical observables can be correct even with virtual degeneracy

### Problem 2: Gauge Redundancy

- The degenerate virtual eigenvalues create a **manifold of equivalent solutions**
- All solutions on this manifold give the same physical observables
- Regularization on physical quantities cannot select a unique point on this manifold

### Problem 3: Local Minima

- The optimizer (CMA-ES) finds a solution satisfying the energy constraint
- Many gauge-equivalent solutions exist with similar energies
- No gradient drives the system away from the degenerate subspace

---

## Implications

### 1. The IsoPEPS Ansatz is Over-Parameterized

For the parameter regime tested (g=0.5 to g=2.0, row=3), the isometric PEPS with bond dimension D=2 appears to have **more parameters than necessary**:

- The physical ground state may be representable with fewer parameters
- The extra parameters manifest as gauge freedom
- This is not necessarily a bug — it means the ansatz is expressive enough

### 2. Physical Observables Remain Correct

Despite the virtual degeneracy:

- Energy estimates are accurate (match PEPSKit reference)
- Expectation values ⟨X⟩, ⟨Z⟩, ⟨ZZ⟩ are computed correctly
- The **physical channel analysis correctly identifies** the product-state nature

### 3. Entropy Computation Requires Care

- `multiline_mps_entanglement` (virtual) gives **large entropy** due to gauge redundancy
- `multiline_mps_physical_entanglement_infinite` (physical) gives **correct small entropy**
- **Always use physical channel** for interpreting entanglement of the quantum state

---

## Recommendations

### Short-term Fixes

1. **Use physical channel metrics** for analysis (already implemented)
2. **Report both** virtual and physical entropies to identify gauge issues
3. **Large physical gap + small physical entropy** = healthy product-like state

### Potential Improvements

1. **Gauge Fixing**: Add explicit gauge-fixing constraints to select a canonical form
   
2. **Canonical Form Regularization**: Penalize deviation from left/right canonical MPS form
   
3. **Virtual Space Regularization**: Add penalty on virtual transfer matrix gap:
   ```julia
   cost = energy + gap_weight * (1.0 - virtual_gap)
   ```
   
4. **Different Ansatz**: For product-state-like regimes, consider simpler ansätze

### When is Degeneracy Expected?

Virtual degeneracy is most likely when:

- Physical state has **area-law entanglement** (gapped phase)
- Bond dimension D is **larger than needed**
- System is **far from critical point** (g << g_c or g >> g_c)

---

## Conclusion

The observed virtual transfer matrix degeneracy is a **feature, not a bug** of the isometric PEPS ansatz when applied to simple (product-like) quantum states. The key insight is:

> **Physical observables and physical entanglement are correctly captured**, even when the virtual representation has gauge redundancy.

The `multiline_mps_physical_entanglement_infinite` function correctly computes the physical entanglement by using the physical channel fixed point, resolving the ambiguity caused by virtual degeneracy.

---

## References

- Related notes: `product_state_problem_and_solutions.md`, `small_gap.md`
- Code: `src/entanglement.jl`, `src/transfer_matrix.jl`

