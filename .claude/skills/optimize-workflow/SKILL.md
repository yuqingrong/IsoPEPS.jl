# optimize-workflow

Set up and run optimization workflows for quantum state training.

## Usage
```
/optimize-workflow <task> [options]
```

## Tasks
- `setup`: Create new optimization configuration
- `run`: Execute optimization with given parameters
- `resume`: Resume interrupted optimization
- `analyze`: Analyze optimization results

## Options
- `--optimizer <name>`: Choose optimizer (cmaes, optim, gradient)
- `--max-iter <n>`: Maximum iterations
- `--checkpoint`: Enable checkpointing
- `--parallel`: Use parallel evaluation

## Workflow

### Setup Task

1. **Define optimization problem**
   - Target state or Hamiltonian
   - Initial state parameters
   - Objective function (energy, fidelity, etc.)

2. **Choose optimizer**
   - CMAEvolutionStrategy for derivative-free
   - Optim.jl for gradient-based
   - Custom optimizer configuration

3. **Configure parameters**
   - Population size (for CMA-ES)
   - Learning rate (for gradient methods)
   - Convergence criteria
   - Checkpoint frequency

4. **Create configuration file**
   - Save as JSON or TOML
   - Include all parameters
   - Document choices

### Run Task

1. **Load configuration**
   - Parse configuration file
   - Validate parameters
   - Initialize optimizer

2. **Setup monitoring**
   - Create log file
   - Setup progress callbacks
   - Configure checkpointing

3. **Execute optimization**
   - Run optimization loop
   - Log progress at intervals
   - Save checkpoints
   - Handle interruptions gracefully

4. **Save results**
   - Final parameters
   - Convergence history
   - Performance metrics
   - Visualization data

### Resume Task

1. **Load checkpoint**
   - Find latest checkpoint
   - Restore optimizer state
   - Verify integrity

2. **Continue optimization**
   - Resume from saved state
   - Update configuration if needed
   - Continue logging

### Analyze Task

1. **Load results**
   - Read optimization history
   - Load final state

2. **Generate analysis**
   - Convergence plots
   - Energy/fidelity evolution
   - Parameter distributions
   - Correlation analysis

3. **Create report**
   - Summary statistics
   - Visualizations
   - Recommendations

## Example Configuration

```julia
config = Dict(
    "optimizer" => "cmaes",
    "max_iterations" => 1000,
    "population_size" => 20,
    "initial_sigma" => 0.1,
    "target" => "ground_state",
    "hamiltonian" => "heisenberg",
    "checkpoint_every" => 50,
    "convergence_tol" => 1e-6
)
```

## Example Usage

```julia
# Setup
/optimize-workflow setup --optimizer cmaes --max-iter 1000

# Run
/optimize-workflow run --checkpoint --parallel

# Resume after interruption
/optimize-workflow resume

# Analyze results
/optimize-workflow analyze
```

## Success Criteria
- Optimization converges or reaches max iterations
- Checkpoints are saved correctly
- Results are reproducible
- Analysis provides actionable insights
