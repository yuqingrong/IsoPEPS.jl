# benchmark

Run performance benchmarks for IsoPEPS.jl operations.

## Usage
```
/benchmark [target] [options]
```

## Options
- `target`: Specific function or module to benchmark (default: all)
- `--compare`: Compare with previous benchmark results
- `--profile`: Generate detailed profiling data
- `--save`: Save results to file

## Workflow

1. **Setup benchmarking environment**
   - Check for BenchmarkTools.jl dependency
   - Create benchmark directory if needed
   - Load previous results if comparing

2. **Identify benchmark targets**
   - Tensor contractions (OMEinsum operations)
   - Gate applications
   - Observable computations
   - Training iterations
   - Transfer matrix operations

3. **Run benchmarks**
   - Use `@benchmark` macro for timing
   - Collect statistics (min, median, max, memory)
   - Run multiple samples for reliability
   - Measure memory allocations

4. **Analyze results**
   - Identify performance bottlenecks
   - Compare with previous runs (if available)
   - Flag regressions
   - Suggest optimizations

5. **Generate report**
   - Summary table with timing and memory
   - Comparison charts (if comparing)
   - Recommendations for optimization

## Example Benchmarks

### Tensor Contraction
```julia
using BenchmarkTools
@benchmark ein"ij,jk,kl->il"(A, B, C)
```

### Gate Application
```julia
@benchmark apply_gate!(state, gate, site)
```

### Observable Measurement
```julia
@benchmark compute_energy(state, hamiltonian)
```

## Report Format
```
IsoPEPS.jl Benchmark Results
============================

Tensor Operations:
  ein"ij,jk->ik" (10×10):  12.3 μs (median)  |  480 bytes
  ein"ijk,jkl->il" (5×5×5): 45.7 μs (median) | 1.2 KB

Gate Applications:
  apply_gate! (single):     8.2 μs (median)  |  320 bytes
  apply_gate! (two-qubit): 23.1 μs (median)  |  640 bytes

Observables:
  energy:                  156.4 μs (median) | 2.1 KB
  magnetization:            34.2 μs (median) |  512 bytes

Training:
  optimization_step:        2.3 ms (median)  | 45 KB

Recommendations:
  - Consider optimizing tensor contraction order in energy computation
  - Gate application shows good performance
```

## Success Criteria
- Benchmarks complete without errors
- Results are reproducible
- Memory allocations are tracked
- Report is generated
