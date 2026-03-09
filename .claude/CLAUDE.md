# CLAUDE.md

## Project Overview
Julia package for isometric Projected Entangled Pair States (IsoPEPS) - a tensor network approach for quantum many-body systems. Implements quantum state representation, optimization, and visualization using tensor network methods.

## Skills
- [test-runner](skills/test-runner/SKILL.md) -- Run Julia tests with coverage reporting and detailed output
- [add-gate](skills/add-gate/SKILL.md) -- Add a new quantum gate implementation to the package
- [add-observable](skills/add-observable/SKILL.md) -- Add a new observable measurement to the package
- [benchmark](skills/benchmark/SKILL.md) -- Run performance benchmarks for tensor operations and quantum computations
- [optimize-workflow](skills/optimize-workflow/SKILL.md) -- Set up and run optimization workflows for quantum state training
- [visualize](skills/visualize/SKILL.md) -- Generate plots and visualizations for quantum states and results
- [release](skills/release/SKILL.md) -- Create a new package release with version management

## Architecture

### Core Modules
- `src/IsoPEPS.jl` - Main module and exports
- `src/gates.jl` - Quantum gate implementations
- `src/observables.jl` - Observable measurements and expectation values
- `src/quantum_channels.jl` - Quantum channel operations
- `src/reference.jl` - Reference implementations
- `src/training.jl` - Training and optimization routines
- `src/transfer_matrix.jl` - Transfer matrix methods
- `src/visualization.jl` - Plotting and visualization utilities

### Key Dependencies
- **Tensor Networks**: ITensors, ITensorMPS, OMEinsum
- **Quantum Computing**: Yao, YaoBlocks
- **Optimization**: Optim, Optimization, CMAEvolutionStrategy
- **Visualization**: CairoMakie
- **Linear Algebra**: KrylovKit, LinearAlgebra

### Extensions
- `ITensorsExt` - ITensors and ITensorInfiniteMPS integration
- `MPSKitExt` - MPSKit and TensorKit integration
- `ManifoldsManoptExt` - Manifold optimization
- `PEPSKitExt` - PEPSKit integration

## Commands
```bash
julia --project=. -e 'using Pkg; Pkg.test()'        # Run tests
julia --project=. -e 'using Pkg; Pkg.instantiate()' # Install dependencies
julia --project=.                                     # Start REPL with project
```

## Git Safety
- **NEVER force push** (`git push --force`, `git push -f`, `git push --force-with-lease`)
- Always create new commits rather than amending published commits
- Never skip hooks (--no-verify) unless explicitly requested

## Conventions

### File Naming
- Module files: lowercase with underscores (e.g., `transfer_matrix.jl`)
- Test files: `test/runtests.jl` and subdirectories
- Example files: `examples/` directory

### Code Style
- Follow Julia standard style guide
- Use 4 spaces for indentation
- Type annotations for function arguments when clarity is needed
- Docstrings for exported functions using Julia's docstring format
- Use `@doc` for documentation

### Testing Requirements
- Unit tests in `test/` directory
- Use `@testset` for organizing tests
- Test coverage for new features
- Integration tests for complex workflows

## Documentation Locations
- `README.md` - Project overview and badges
- `.claude/` - Claude Code instructions
- Inline docstrings - Function and type documentation
- `docs/` - Extended documentation (if exists)

## Development Workflow
1. Create feature branch from `master`
2. Implement changes with tests
3. Run test suite locally
4. Commit with descriptive messages
5. Create PR to `master` branch
6. Address review comments
7. Merge after CI passes

## Key Patterns
- Tensor network operations use OMEinsum for efficient contractions
- Quantum gates implemented using Yao framework
- Optimization uses multiple backends (Optim, CMAEvolutionStrategy)
- Visualization with CairoMakie for publication-quality figures
- Extension system for optional heavy dependencies

## Performance Considerations
- Use in-place operations where possible
- Leverage OMEinsum contraction order optimization
- Consider memory allocation in hot loops
- Profile before optimizing

## Current Branch
- Working on: `trainoptimization`
- Main branch: `master`
- Status: clean working directory
