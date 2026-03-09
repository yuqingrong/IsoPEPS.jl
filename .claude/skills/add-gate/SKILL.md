# add-gate

Add a new quantum gate implementation to the IsoPEPS.jl package.

## Usage
```
/add-gate <gate_name> [options]
```

## Options
- `--parametric`: Create a parametric gate (e.g., rotation gates)
- `--multi-qubit`: Multi-qubit gate (default: single-qubit)
- `--with-tests`: Generate test cases automatically

## Workflow

1. **Gather requirements**
   - Gate name and type (unitary, parametric, etc.)
   - Number of qubits
   - Mathematical definition (matrix representation)
   - Physical interpretation

2. **Check existing gates**
   - Read `src/gates.jl`
   - Ensure no naming conflicts
   - Identify similar gates for reference

3. **Implement gate**
   - Add gate function to `src/gates.jl`
   - Include docstring with:
     - Mathematical definition
     - Parameters (if parametric)
     - Usage examples
   - Implement using Yao framework
   - Add matrix representation

4. **Create tests**
   - Add test cases in `test/gates_test.jl` (or create if needed)
   - Test unitarity
   - Test known eigenvalues/eigenvectors
   - Test composition with other gates
   - Test edge cases for parametric gates

5. **Update exports**
   - Add to export list in `src/IsoPEPS.jl`

6. **Documentation**
   - Add usage example
   - Document any special properties

## Example

For a parametric rotation gate:
```julia
"""
    rx_gate(θ::Real)

Rotation gate around X-axis by angle θ.

# Arguments
- `θ::Real`: Rotation angle in radians

# Returns
- Yao gate block for Rx(θ)

# Examples
```jldoctest
julia> gate = rx_gate(π/4)
julia> apply!(state, gate)
```
"""
function rx_gate(θ::Real)
    return Rx(θ)
end
```

## Success Criteria
- Gate correctly implements the mathematical definition
- All tests pass
- Documentation is complete
- Exports are updated
