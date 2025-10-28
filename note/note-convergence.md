To determine the thermalization time `m` for your Markov chain sampler, here are several practical methods:

## 1. **Autocorrelation Analysis** (Most Common)
Calculate the autocorrelation function of your observable:
```julia
function autocorrelation(data, max_lag)
    n = length(data)
    mean_data = mean(data)
    var_data = var(data)
    
    autocorr = zeros(max_lag)
    for lag in 1:max_lag
        autocorr[lag] = sum((data[1:n-lag] .- mean_data) .* 
                           (data[lag+1:n] .- mean_data)) / (n - lag) / var_data
    end
    return autocorr
end

# Integrated autocorrelation time
τ_int = 1 + 2 * sum(autocorr[autocorr .> 0])  # Sum until correlation drops
m_thermalization = 5 * τ_int  # Rule of thumb: 5-10 times τ_int
```

## 2. **Visual Inspection of Observables**
Plot key observables (energy, magnetization, etc.) vs Monte Carlo steps:
```julia
# Track an observable during sampling
observable = [compute_observable(bitstring) for bitstring in samples]
plot(observable)
# m is where the plot "settles" into equilibrium fluctuations
```

## 3. **Multiple Independent Chains (Gelman-Rubin)**
Run multiple chains from different initial conditions:
```julia
function gelman_rubin(chains)
    # chains: matrix where each column is a chain
    n_chains, n_steps = size(chains)
    
    # Between-chain variance
    chain_means = mean(chains, dims=2)
    B = n_steps * var(chain_means)
    
    # Within-chain variance
    W = mean(var(chains, dims=2))
    
    # Potential scale reduction factor
    R̂ = sqrt((W + B/n_steps) / W)
    return R̂  # Should be < 1.1 for convergence
end
```

## 4. **Running Average Convergence**
Check when running averages stabilize:
```julia
function running_average(data)
    cumsum(data) ./ (1:length(data))
end

# m is where derivative becomes small
running_avg = running_average(observable)
m = findfirst(abs.(diff(running_avg)) .< threshold)
```

## 5. **Energy/Hamiltonian Criterion** (For Physics Problems)
If sampling from a physical system:
```julia
# Track energy convergence
energies = [compute_energy(bitstring) for bitstring in samples]
# m is where mean(energies[m:end]) ≈ theoretical_value
```

## Practical Recommendations:

**Quick check approach:**
```julia
# 1. Run long chain
n_total = 100000
samples = run_mcmc(n_total)

# 2. Compute observable
obs = [measure(s) for s in samples]

# 3. Split into blocks and check variance
block_size = 1000
n_blocks = 10
blocks = [mean(obs[i*block_size:(i+1)*block_size]) for i in 0:n_blocks-1]

# 4. When block variance stabilizes, you've thermalized
plot([var(blocks[1:k]) for k in 2:n_blocks])
```

**Conservative choice:** Use `m = 10 × τ_int` where `τ_int` is the integrated autocorrelation time.

Would you like me to look at your specific sampler code to implement one of these methods? I can check your `iPEPS.jl` file if you're implementing sampling there.