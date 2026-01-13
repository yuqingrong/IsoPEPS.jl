# Using tmux to Run Multiple Simulations

## Quick Start

### 1. Start tmux
```bash
cd /home/yuqingrong/IsoPEPS.jl
tmux
```

### 2. Split into panes
- Split horizontally: `Ctrl+b` then `"`
- Split vertically: `Ctrl+b` then `%`
- Navigate between panes: `Ctrl+b` then arrow keys

### 3. In each pane, run a different simulation

**Pane 1 (row=3):**
```bash
julia --project --threads=44 -e '
using IsoPEPS, Optimization, OptimizationCMAEvolutionStrategy, Random, Yao
Random.seed!(12)
params = ones(40)  # 2*5*4 = 40 params for p=4, nqubits=5
result = optimize_circuit(params, 1.0, 2.0, 4, 3, 5; maxiter=5000, measure_first=:Z)
println("Row 3 completed! Energy: ", result.energy)
'
```

**Pane 2 (row=4):**
```bash
julia --project --threads=44 -e '
using IsoPEPS, Optimization, OptimizationCMAEvolutionStrategy, Random, Yao
Random.seed!(13)
params = ones(40)
result = optimize_circuit(params, 1.0, 2.0, 4, 4, 5; maxiter=5000, measure_first=:Z)
println("Row 4 completed! Energy: ", result.energy)
'
```

**Pane 3 (row=5):**
```bash
julia --project --threads=44 -e '
using IsoPEPS, Optimization, OptimizationCMAEvolutionStrategy, Random, Yao
Random.seed!(14)
params = ones(40)
result = optimize_circuit(params, 1.0, 2.0, 4, 5, 5; maxiter=5000, measure_first=:Z)
println("Row 5 completed! Energy: ", result.energy)
'
```

**Pane 4 (row=6):**
```bash
julia --project --threads=44 -e '
using IsoPEPS, Optimization, OptimizationCMAEvolutionStrategy, Random, Yao
Random.seed!(15)
params = ones(40)
result = optimize_circuit(params, 1.0, 2.0, 4, 6, 5; maxiter=5000, measure_first=:Z)
println("Row 6 completed! Energy: ", result.energy)
'
```

## Even Simpler: Use separate script files

Just edit `project/simulation.jl` to change the parameters at the bottom, then:

**In tmux pane 1:**
```bash
# Edit simulation.jl to set row=3
julia --project --threads=44 project/simulation.jl
```

**In tmux pane 2:**
```bash
# Edit to set row=4, run in different folder or rename output
julia --project --threads=44 project/simulation.jl
```

## tmux Quick Reference

| Command | Action |
|---------|--------|
| `Ctrl+b` then `"` | Split pane horizontally |
| `Ctrl+b` then `%` | Split pane vertically |
| `Ctrl+b` then arrow | Navigate panes |
| `Ctrl+b` then `x` | Kill current pane |
| `Ctrl+b` then `d` | Detach session |
| `tmux attach` | Reattach to session |
| `Ctrl+b` then `[` | Enter scroll mode (q to exit) |
| `exit` | Close current pane |

## Detach and Check Later

1. Start all simulations in tmux
2. Press `Ctrl+b` then `d` to detach
3. Close your terminal (simulations keep running!)
4. Later, reconnect: `tmux attach`
