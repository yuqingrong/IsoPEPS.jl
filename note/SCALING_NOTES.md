# Scaling Quantum Simulations Beyond Exact State Vectors

## Current Implementation: Adaptive Exact Simulation
- **Method**: Full state vector simulation with adaptive sampling
- **Limit**: ~20-22 qubits (16-64 MB memory)
- **Status**: ✅ Implemented

## Future Improvements for Larger Systems

### 1. Matrix Product State (MPS) Representation
- **What**: Compress quantum state using tensor networks
- **Benefit**: Exponential → Polynomial memory scaling
- **Limit**: Can handle 50-100 qubits
- **Packages**: ITensors.jl (already in dependencies!)

### 2. GPU Acceleration
- **What**: Use GPU for state vector operations
- **Benefit**: 10-100x speedup for large state vectors
- **Limit**: GPU memory dependent (~24-48 GB on modern GPUs)
- **Packages**: CUDA.jl, YaoArrayRegister.jl with GPU backend

### 3. Clifford Circuit Optimization
- **What**: Use stabilizer formalism for Clifford gates
- **Benefit**: Polynomial scaling for certain gate types
- **Limit**: Only works for specific quantum channels
- **Packages**: QuantumClifford.jl

### 4. Approximate Sampling Methods
- **What**: Monte Carlo sampling without full state
- **Benefit**: Avoid storing full quantum state
- **Limit**: Statistical noise, less accurate
- **Approach**: Stochastic wavefunction methods

### 5. Distributed Computing
- **What**: Split computation across multiple nodes
- **Benefit**: Aggregate memory from multiple machines
- **Limit**: Communication overhead
- **Packages**: MPI.jl, Distributed.jl

## Recommended Progression

1. **Current (row ≤ 8)**: Use adaptive exact simulation ✅
2. **Medium (row 9-15)**: Switch to MPS/tensor networks
3. **Large (row 16+)**: GPU acceleration + tensor networks
4. **Very large (row 20+)**: Distributed tensor networks or approximate methods

## Implementation Priority

For IsoPEPS.jl, next steps should be:

1. ✅ **Adaptive sampling** (Done!)
2. **MPS backend** using ITensors.jl (already available)
3. **GPU support** for state vectors (moderate effort)
4. **Benchmark suite** to compare methods
