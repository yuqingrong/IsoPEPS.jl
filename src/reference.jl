"""
    mpskit_ground_state(d, D, g, row)

Compute exact ground state using MPSKit (VUMPS algorithm).

# Arguments
- `d`: Physical dimension (usually 2)
- `D`: Bond dimension
- `g`: Transverse field strength
- `row`: Number of rows (cylinder circumference)

# Returns
Named tuple with:
- `energy`: Ground state energy per site
- `correlation_length`: Correlation length
- `entropy`: Entanglement entropy
- `spectrum`: Transfer matrix spectrum

# Description
Uses VUMPS from MPSKit to find the ground state of the transverse field 
Ising model on an infinite cylinder: H = -g∑Xᵢ - ∑ZᵢZⱼ
"""
function mpskit_ground_state(d::Int, D::Int, g::Float64, row::Int)
    mps = InfiniteMPS([ComplexSpace(d) for _ in 1:row], [ComplexSpace(D) for _ in 1:row])
    H = transverse_field_ising(InfiniteCylinder(row); g=g)
    psi, _ = find_groundstate(mps, H, VUMPS())
    
    E = real(expectation_value(psi, H)) / row
    spectrum = transfer_spectrum(psi)
    corr_lengths = correlation_length(psi)
    
    len = isempty(corr_lengths) ? NaN : corr_lengths[1]
    entropy = MPSKit.entropy(psi)
    
    return (energy=E, correlation_length=len, entropy=entropy, spectrum=spectrum)
end

"""
    mpskit_ground_state_1d(d, D, g)

Compute 1D chain ground state using MPSKit.

# Arguments
- `d`: Physical dimension
- `D`: Bond dimension
- `g`: Transverse field strength

# Returns
Named tuple with energy, correlation_length, entropy, spectrum, and psi (wavefunction)
"""
function mpskit_ground_state_1d(d::Int, D::Int, g::Float64)
    mps = InfiniteMPS([ComplexSpace(d)], [ComplexSpace(D)])
    H = transverse_field_ising(; g=g)
    psi, _ = find_groundstate(mps, H, VUMPS())
    
    E = real(expectation_value(psi, H))
    spectrum = transfer_spectrum(psi)
    corr_lengths = correlation_length(psi)
    
    len = isempty(corr_lengths) ? NaN : corr_lengths[1]
    entropy = MPSKit.entropy(psi)
    
    return (energy=E, correlation_length=len, entropy=entropy, spectrum=spectrum, psi=psi)
end

"""
    pepskit_ground_state(d, D, J, g; χ=20, ctmrg_tol=1e-10, grad_tol=1e-6, maxiter=1000)

Compute 2D PEPS ground state using PEPSKit.

# Arguments
- `d`: Physical dimension
- `D`: PEPS bond dimension
- `J`: Coupling strength
- `g`: Transverse field strength
- `χ`: Environment bond dimension for CTMRG (default: 20)
- `ctmrg_tol`: CTMRG convergence tolerance (default: 1e-10)
- `grad_tol`: Gradient tolerance (default: 1e-6)
- `maxiter`: Maximum iterations (default: 1000)

# Returns
Ground state energy per site
"""
function pepskit_ground_state(d::Int, D::Int, J::Float64, g::Float64; 
                               χ::Int=20, ctmrg_tol::Float64=1e-10, 
                               grad_tol::Float64=1e-6, maxiter::Int=1000)
    H = transverse_field_ising(PEPSKit.InfiniteSquare(); g=g)
    peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χ)), peps₀; tol=ctmrg_tol)
    
    peps, env, E, = fixedpoint(H, peps₀, env₀; 
                                tol=grad_tol, 
                                boundary_alg=(; tol=ctmrg_tol), 
                                optimizer_alg=(; maxiter=maxiter))
    return E
end
