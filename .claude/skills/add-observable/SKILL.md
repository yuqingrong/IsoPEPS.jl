# add-observable

Add a new observable measurement to the IsoPEPS.jl package.

## Usage
```
/add-observable <observable_name> [options]
```

## Options
- `--local`: Local observable (single-site)
- `--correlation`: Two-point correlation function
- `--global`: Global observable (system-wide)
- `--with-tests`: Generate test cases automatically

## Workflow

1. **Gather requirements**
   - Observable name and type
   - Physical quantity being measured
   - Mathematical definition
   - Expected value range

2. **Check existing observables**
   - Read `src/observables.jl`
   - Ensure no naming conflicts
   - Identify similar observables for reference

3. **Implement observable**
   - Add function to `src/observables.jl`
   - Include comprehensive docstring:
     - Physical interpretation
     - Mathematical formula
     - Parameters and return type
     - Usage examples
   - Implement measurement logic
   - Handle different state representations (MPS, PEPS, etc.)

4. **Create tests**
   - Add test cases in `test/observables_test.jl`
   - Test against known analytical results
   - Test numerical stability
   - Test edge cases (zero state, maximally entangled, etc.)
   - Verify hermiticity if applicable

5. **Update exports**
   - Add to export list in `src/IsoPEPS.jl`

6. **Documentation**
   - Add usage examples
   - Document physical interpretation
   - Note any computational complexity considerations

## Example

For a magnetization observable:
```julia
"""
    magnetization_z(state, site::Int)

Compute the z-component of magnetization at a given site.

# Arguments
- `state`: Quantum state (MPS or PEPS)
- `site::Int`: Site index

# Returns
- `Float64`: Expectation value ⟨σᶻ⟩ ∈ [-1, 1]

# Examples
```jldoctest
julia> mz = magnetization_z(ground_state, 5)
0.342
```
"""
function magnetization_z(state, site::Int)
    σz = [1 0; 0 -1]
    return expect(state, σz, site)
end
```

## Success Criteria
- Observable correctly computes expectation values
- Results match analytical values for test cases
- All tests pass
- Documentation is complete
- Physical interpretation is clear
