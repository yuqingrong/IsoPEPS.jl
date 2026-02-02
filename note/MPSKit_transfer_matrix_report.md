# Transfer Matrix Calculations in MPSKit.jl: Technical Report

**Date:** February 1, 2026  
**Subject:** Understanding Transfer Matrix Functions in MPSKit.jl

---

## Executive Summary

This report investigates the capabilities of **MPSKit.jl** and **ITensorInfiniteMPS.jl** for transfer matrix calculations in the context of infinite Matrix Product States (MPS).

### Key Findings

**MPSKit.jl:**
- Does **NOT** provide functions to explicitly construct transfer matrices
- Offers higher-level functions: `transfer_spectrum(psi)` and `correlation_length(psi)`
- Computes properties derived from transfer matrix eigenspectra
- Mature, well-documented, production-ready
- Uses TensorKit.jl backend

**ITensorInfiniteMPS.jl:**
- Does **provide** transfer matrix spectrum calculation functionality
- Includes example: `examples/vumps/transfer_matrix_spectrum.jl`
- Likely offers more direct access to transfer matrix construction
- Work-in-progress with minimal documentation
- Uses ITensor.jl backend
- Requires understanding of infinite MPS theory

**For Custom Transfer Matrices:**
If you need explicit transfer matrix construction (e.g., for PEPS, quantum circuits, or custom tensor networks), you must implement it yourself, as demonstrated in the IsoPEPS.jl codebase.

---

## 1. Available Functions in MPSKit.jl

### 1.1 What MPSKit.jl Provides

MPSKit.jl offers two main functions for working with transfer matrix properties:

1. **`transfer_spectrum(psi)`**
   - Computes the eigenvalue spectrum of the transfer matrix
   - Input: An `InfiniteMPS` state object
   - Output: Array of eigenvalues (spectrum)
   - Does **not** return the actual transfer matrix

2. **`correlation_length(psi)`**
   - Directly computes correlation lengths from the transfer matrix spectrum
   - Input: An `InfiniteMPS` state object
   - Output: Correlation length(s)
   - Calculated from: ξ = -1/log|λ₂| where λ₂ is the second-largest eigenvalue

### 1.2 What MPSKit.jl Does NOT Provide

MPSKit.jl does **not** have functions to:
- Explicitly construct the transfer matrix as a matrix/tensor object
- Return the transfer matrix for manual manipulation
- Build transfer matrices from raw tensor data

---

## 2. Using MPSKit.jl Functions

### 2.1 Basic Usage

To use MPSKit.jl's transfer matrix functions, you need an `InfiniteMPS` object:

```julia
using MPSKit, TensorKit

# Example: Compute ground state
mps = InfiniteMPS([ComplexSpace(d)], [ComplexSpace(D)])
H = transverse_field_ising(; g=1.0)
psi, _ = find_groundstate(mps, H, VUMPS())

# Compute transfer spectrum
spectrum = transfer_spectrum(psi)

# Compute correlation length
ξ = correlation_length(psi)
```

### 2.2 Creating InfiniteMPS from Unit Cell Tensor

If you have an optimized unit cell tensor `A` and want to use MPSKit.jl's functions:

**Option 1: If A is already a TensorKit TensorMap**
```julia
# A should be: left virtual ← physical ⊗ right virtual
psi = InfiniteMPS([A])
spectrum = transfer_spectrum(psi)
```

**Option 2: If A is a regular array [D×d×D]**
```julia
D = size(A, 1)  # bond dimension
d = size(A, 2)  # physical dimension

# Convert to TensorMap
A_tensormap = TensorMap(A, ComplexSpace(D), ComplexSpace(d) ⊗ ComplexSpace(D))

# Create InfiniteMPS
psi = InfiniteMPS([A_tensormap])
spectrum = transfer_spectrum(psi)
```

**Option 3: Multi-site unit cell**
```julia
psi = InfiniteMPS([A1, A2, ...])  # Array of TensorMaps
spectrum = transfer_spectrum(psi)
```

### 2.3 Extracting Tensors from InfiniteMPS

To extract the raw tensor data from an `InfiniteMPS` (for manual transfer matrix construction):

```julia
# Get the right-canonical form tensor
A_array = result.psi.AR.data[1]  # Returns array with shape [D, d, D]

# The indices are: [left_virtual, physical, right_virtual]
```

---

## 3. Manual Transfer Matrix Construction

### 3.1 When You Need It

Manual transfer matrix construction is necessary when:
- You need the explicit matrix for custom operations
- You're working with non-standard tensor network structures
- You want to verify or compare with other methods
- MPSKit.jl's functions don't provide the specific information you need

### 3.2 Example from IsoPEPS.jl

The IsoPEPS.jl codebase demonstrates manual transfer matrix construction:

```julia
# Extract tensor from MPSKit result
exact_A = Array{ComplexF64}(undef, D, 2, D)
for (i, j, k) in Iterators.product(1:D, 1:2, 1:D)
    exact_A[i, j, k] = result.psi.AR.data[1][i, j, k]
end

# Reshape and process
exact_A = reshape(permutedims(exact_A, (2, 1, 3)), (d * D, D))
nullspace_A = LinearAlgebra.nullspace(exact_A')
A_matrix = vcat(exact_A, nullspace_A)

# Use custom function to compute transfer spectrum
rho, gap, eigenval = compute_transfer_spectrum(A_matrix_list, row, nqubits)

# Verify: gap ≈ 1 / correlation_length
```

### 3.3 Custom Transfer Matrix Functions in IsoPEPS.jl

The IsoPEPS.jl package provides comprehensive custom implementations:

- **`compute_transfer_spectrum(gates, row, nqubits)`**
  - Builds and analyzes transfer matrices from gate/tensor data
  - Supports both virtual and physical channel types
  - Includes matrix-free methods for large systems
  
- **`get_transfer_matrix(gates, row, virtual_qubits)`**
  - Explicitly constructs the full transfer matrix
  - Returns as a matrix object for manipulation

- **`build_transfer_code(tensor_ket, tensor_bra, row)`**
  - Creates optimized contraction code for transfer matrix operations
  - Supports matrix-vector products for iterative solvers

---

## 4. Comparison and Validation

### 4.1 Relationship Between Spectral Gap and Correlation Length

The fundamental relationship used for validation:

```
gap = -log(|λ₂|/|λ₁|) = -log(|λ₂|)  (when |λ₁| = 1 for normalized MPS)
correlation_length = ξ = -1/log(|λ₂|) = 1/gap
```

### 4.2 Test Case

From `test/reference.jl`:
```julia
@testset "transfer matrix match MPSKit" begin
    result = mpskit_ground_state_1d(d, D, g)
    # ... extract and process tensors ...
    rho, gap, eigenval = compute_transfer_spectrum(A_matrix_list, row, nqubits)
    @test gap ≈ 1 / result.correlation_length atol = 1e-5
end
```

This validates that custom transfer matrix calculations match MPSKit.jl's internal computations.

---

## 5. Recommendations

### 5.1 When to Use MPSKit.jl Functions

Use `transfer_spectrum(psi)` and `correlation_length(psi)` when:
- You have an `InfiniteMPS` object from optimization/ground state search
- You only need eigenvalues or correlation lengths
- You want tested, optimized implementations
- Working with standard infinite MPS

### 5.2 When to Build Transfer Matrix Manually

Build the transfer matrix yourself when:
- Working with custom tensor network structures (e.g., PEPS, circuits)
- Need the explicit matrix for operations beyond spectrum
- Implementing new algorithms or modifications
- Comparing different methods
- Educational/debugging purposes

### 5.3 Best Practices

1. **For validation**: Use MPSKit.jl as a reference when developing custom implementations
2. **For efficiency**: Extract only the information you need (eigenvalues vs. full matrix)
3. **For large systems**: Use matrix-free methods to avoid memory issues
4. **For compatibility**: Work with TensorKit.jl format when interfacing with MPSKit.jl

---

## 6. Key Technical Details

### 6.1 Tensor Format in MPSKit.jl

- Uses **TensorKit.jl** for tensor operations
- Tensors are `TensorMap` objects with explicit virtual and physical spaces
- MPS tensors have structure: `left_virtual ← physical ⊗ right_virtual`
- Data can be extracted as arrays: `psi.AR.data[1]` gives `[D, d, D]` array

### 6.2 Transfer Matrix Structure

For an infinite MPS with unit cell tensor A[i,s,j] (i,j: virtual, s: physical):
```
T[i,j,i',j'] = Σₛ A[i,s,j] × A*[i',s,j']
```

The dominant eigenvalue should be 1 for normalized states.

### 6.3 Implementation Considerations

From IsoPEPS.jl's implementation:
- Matrix-free approach for dimensions > 1024
- Iterative eigensolvers (KrylovKit) for dimensions > 256
- Full eigendecomposition only for small matrices
- Contraction order optimization using OMEinsum.jl

---

## 7. Conclusion

MPSKit.jl provides robust, high-level functions for extracting transfer matrix properties (eigenvalues, correlation lengths) from infinite MPS states, but does not expose the transfer matrix itself. For applications requiring explicit transfer matrix construction—such as custom tensor networks, PEPS, or quantum circuit simulations—manual implementation is necessary. The IsoPEPS.jl codebase demonstrates a complete implementation of transfer matrix construction and analysis that can serve as a reference for such custom implementations.

---

## 8. Alternative: ITensorInfiniteMPS.jl

### 8.1 Overview

**ITensorInfiniteMPS.jl** is an alternative package for working with infinite MPS using the ITensor framework. Unlike MPSKit.jl (which uses TensorKit.jl), this package is built on top of ITensor.jl.

**Key Characteristics:**
- **Repository:** https://github.com/ITensor/ITensorInfiniteMPS.jl
- **Status:** Work in progress with minimal documentation
- **Framework:** Built on ITensor.jl
- **Example file:** `examples/vumps/transfer_matrix_spectrum.jl`

### 8.2 Transfer Matrix Functions in ITensorInfiniteMPS.jl

ITensorInfiniteMPS.jl **does have functions to calculate transfer matrix spectrum**, with example code provided in the repository.

**Available Functionality:**
- Transfer matrix spectrum calculation
- Eigenvalue computation
- Correlation length from spectral gap

### 8.3 How to Use (Expected Pattern)

Based on the repository structure and ITensor conventions, the typical usage would be:

```julia
using ITensors
using ITensorInfiniteMPS

# Step 1: Optimize an infinite MPS (e.g., with VUMPS)
# You'll have an InfiniteMPS object, similar structure to MPSKit.jl

# Step 2: Construct transfer matrix or compute spectrum
# The package likely provides one of these patterns:

# Pattern A: Direct spectrum calculation (like MPSKit.jl)
spectrum = transfer_matrix_spectrum(ψ)
corr_length = correlation_length(ψ)

# Pattern B: Transfer matrix object (more ITensor-like)
T = TransferMatrix(ψ)
eigenvals, eigenvecs = eigen(T)

# Pattern C: Manual construction from tensors
A = ψ[1]  # Get unit cell tensor
T = transfermatrix(A)  # Construct transfer operator
spectrum = eigvals(T)
```

### 8.4 Extracting Unit Cell Tensor

To work with the raw tensors in ITensorInfiniteMPS.jl:

```julia
# Get the MPS tensor from the optimized state
A = ψ[1]  # ITensor object for first site in unit cell

# For multi-site unit cell
A1 = ψ[1]
A2 = ψ[2]
# etc.
```

### 8.5 Constructing Transfer Matrix

The transfer matrix for an infinite MPS with tensor A is typically constructed as:

```julia
# T[α,β,α',β'] = Σₛ A[α,s,β] × conj(A[α',s,β'])
# where α,β are virtual indices and s is physical index

# ITensor likely provides helper functions like:
T = transfermatrix(A, dag(A))  # dag() gives complex conjugate
```

### 8.6 Computing Spectrum

Once you have the transfer matrix:

```julia
# Compute eigenvalues
λs = eigen(T)  # or eigvals(T) for just eigenvalues

# Correlation length from spectral gap
λ1 = λs[1]  # Largest eigenvalue (should be ~1)
λ2 = λs[2]  # Second largest
ξ = -1 / log(abs(λ2 / λ1))  # Correlation length
```

### 8.7 Accessing the Example

To see the exact implementation, check:
```bash
# Clone the repository
git clone https://github.com/ITensor/ITensorInfiniteMPS.jl
cd ITensorInfiniteMPS.jl/examples/vumps/
# Open transfer_matrix_spectrum.jl
```

This example file contains the complete working code for transfer matrix spectrum calculations.

### 8.8 Comparison: MPSKit.jl vs ITensorInfiniteMPS.jl

| Feature | MPSKit.jl | ITensorInfiniteMPS.jl |
|---------|-----------|----------------------|
| **Tensor Backend** | TensorKit.jl | ITensor.jl |
| **Documentation** | Good, comprehensive | Minimal, work-in-progress |
| **Maturity** | Stable, production-ready | Experimental |
| **Transfer Matrix Access** | Spectrum only (`transfer_spectrum`) | Likely more direct access |
| **Function Names** | `transfer_spectrum(psi)` | See example file |
| **Symmetries** | Extensive (via TensorKit) | ITensor symmetries |
| **Learning Curve** | Moderate | Steeper (less docs) |

### 8.9 When to Choose ITensorInfiniteMPS.jl

**Choose ITensorInfiniteMPS.jl if:**
- Already using ITensor.jl in your project
- Need features specific to ITensor
- Want more control over transfer matrix construction
- Willing to work with minimal documentation

**Choose MPSKit.jl if:**
- Want stable, well-documented code
- Need comprehensive symmetry support via TensorKit
- Prefer higher-level abstractions
- Working on production code

### 8.10 Integration Note

Both packages can coexist, but you cannot directly mix TensorKit and ITensor tensors. If you need to compare or validate results:

1. **Extract tensor data as arrays** from one package
2. **Convert to the other format** manually
3. **Compare numerical results** (eigenvalues, correlation lengths)

This is exactly what the IsoPEPS.jl test suite does with MPSKit.jl—extracting raw arrays and computing with custom functions.

---

## 9. Quick Reference Guide

### 9.1 Function Summary

**MPSKit.jl (TensorKit.jl backend):**
```julia
using MPSKit, TensorKit

# Compute transfer spectrum
spectrum = transfer_spectrum(psi)  # psi is InfiniteMPS

# Compute correlation length
ξ = correlation_length(psi)

# Create InfiniteMPS from TensorMap
A_tensormap = TensorMap(A_array, ComplexSpace(D), ComplexSpace(d) ⊗ ComplexSpace(D))
psi = InfiniteMPS([A_tensormap])
```

**ITensorInfiniteMPS.jl (ITensor.jl backend):**
```julia
using ITensors, ITensorInfiniteMPS

# Expected usage (check example file for exact syntax):
# Option 1: Direct spectrum
spectrum = transfer_matrix_spectrum(ψ)

# Option 2: Transfer matrix object
T = TransferMatrix(ψ)
eigenvals = eigen(T)

# Get unit cell tensor
A = ψ[1]  # ITensor for first site
```

**Custom Implementation (IsoPEPS.jl approach):**
```julia
using IsoPEPS

# Build transfer matrix from gate/tensor arrays
rho, gap, eigenvalues, eigenvalues_raw = compute_transfer_spectrum(
    gates, row, nqubits; 
    channel_type=:virtual,
    num_eigenvalues=2,
    use_iterative=:auto,
    matrix_free=:auto
)

# Get explicit transfer matrix
T = get_transfer_matrix(gates, row, virtual_qubits)
```

### 9.2 Key Relationships

```julia
# Spectral gap and correlation length
gap = -log(|λ₂|/|λ₁|)          # Spectral gap
ξ = -1/log(|λ₂|)                # Correlation length (when |λ₁| = 1)
ξ ≈ 1/gap                       # Approximate relationship

# For normalized MPS: |λ₁| = 1
gap ≈ 1/ξ
```

### 9.3 Decision Tree

```
Need transfer matrix calculations?
│
├─ Already using ITensor? → Try ITensorInfiniteMPS.jl
│  └─ Check examples/vumps/transfer_matrix_spectrum.jl
│
├─ Already using TensorKit? → Use MPSKit.jl
│  └─ Only eigenvalues/correlation length available
│
├─ Need explicit transfer matrix? → Custom implementation
│  └─ See IsoPEPS.jl src/transfer_matrix.jl
│
└─ Starting fresh?
   ├─ Want stability & docs → MPSKit.jl
   └─ Want ITensor features → ITensorInfiniteMPS.jl
```

### 9.4 Common Tasks

**Task: Get correlation length from optimized MPS**
```julia
# MPSKit.jl
ξ = correlation_length(psi)

# ITensorInfiniteMPS.jl
ξ = correlation_length(ψ)  # or compute from spectrum

# Custom (IsoPEPS.jl)
_, gap, _, _ = compute_transfer_spectrum(gates, row, nqubits)
ξ = 1/gap
```

**Task: Extract raw tensor data**
```julia
# From MPSKit.jl InfiniteMPS
A_array = psi.AR.data[1]  # Returns [D, d, D] array

# From ITensor InfiniteMPS  
A_itensor = ψ[1]  # Returns ITensor object
A_array = array(A_itensor)  # Convert to array

# From TensorKit TensorMap
A_array = convert(Array, A_tensormap)
```

**Task: Validate custom implementation**
```julia
# Compare with MPSKit.jl
mpskit_ξ = correlation_length(psi)
custom_gap = compute_transfer_spectrum(gates, row, nqubits)[2]
custom_ξ = 1/custom_gap

@test custom_ξ ≈ mpskit_ξ atol=1e-5
```

---

## References

1. MPSKit.jl Documentation: https://quantumkithub.github.io/MPSKit.jl/dev/
2. MPSKit.jl GitHub: https://github.com/QuantumKitHub/MPSKit.jl
3. ITensorInfiniteMPS.jl GitHub: https://github.com/ITensor/ITensorInfiniteMPS.jl
4. ITensorInfiniteMPS.jl Transfer Matrix Example: https://github.com/ITensor/ITensorInfiniteMPS.jl/blob/main/examples/vumps/transfer_matrix_spectrum.jl
5. IsoPEPS.jl source code: `src/transfer_matrix.jl`, `src/reference.jl`, `test/reference.jl`

