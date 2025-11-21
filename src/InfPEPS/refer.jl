"""
    result_MPSKit(d::Int, D::Int, g::Float64, J::Float64, row::Int)
    Compute exact ground state energy using MPSKit.jl.

# Arguments
- `d`: Physical dimension
- `D`: Bond dimension
- `g`: Transverse field strength
- `row`: Number of rows (for cylinder)

# Returns
Ground state energy density

# Description
Uses VUMPS algorithm from MPSKit to find the ground state of the
transverse field Ising model on an infinite cylinder.


H = -g*sum_i X_i - J*sum_i Z_iZ_(i+1) - J*sum_i Z_i Z_(i+row)

"""
function result_MPSKit(d::Int, D::Int, g::Float64, row::Int)
    mps = InfiniteMPS([ComplexSpace(d) for _ in 1:row], [ComplexSpace(D) for _ in 1:row])
    H0 = transverse_field_ising(InfiniteCylinder(row); g=g)
    psi, _ = find_groundstate(mps, H0, VUMPS())
    E = real(expectation_value(psi, H0)) / row
    spectrum = transfer_spectrum(psi)
    corr_lengths = correlation_length(psi)
    if isempty(corr_lengths)
        @warn "Correlation length calculation failed (empty result). Using NaN."
        len_gapped = NaN
    else
        len_gapped = corr_lengths[1]
    end
    
    entrop_gapped = MPSKit.entropy(psi)
    return E, len_gapped, entrop_gapped, spectrum
end

function result_1d(d::Int, D::Int, g::Float64)
    mps = InfiniteMPS([ComplexSpace(d)], [ComplexSpace(D)])
    H0 = transverse_field_ising(; g=g)
    psi, _ = find_groundstate(mps, H0, VUMPS())
    E = real(expectation_value(psi, H0))
    spectrum = transfer_spectrum(psi)
    corr_lengths = correlation_length(psi)
    if isempty(corr_lengths)
        @warn "Correlation length calculation failed (empty result). Using NaN."
        len_gapped = NaN
    else
        len_gapped = corr_lengths[1]
    end
    
    entrop_gapped = MPSKit.entropy(psi)
    return E, len_gapped, entrop_gapped, spectrum
end
"""
    result_PEPSKit(d::Int, D::Int, J::Float64, g::Float64; χ ::Int= 20, ctmrg_tol::Float64= 1e-10, grad_tol::Float64= 1e-4, maxiter::Int=1000)
 Compute exact ground state energy using PEPSKit.jl.

# Arguments
- `d`: Physical dimension
- `D`: Bond dimension
- `J`: Coupling strength
- `g`: Transverse field strength
- `χ`: Environment bond dimension for CTMRG (default: 20)
- `ctmrg_tol`: CTMRG convergence tolerance (default: 1e-10)
- `grad_tol`: Gradient tolerance for optimization (default: 1e-6)
- `maxiter`: Maximum optimization iterations (default: 1000)

# Returns
Ground state energy density
"""
function result_PEPSKit(d::Int, D::Int, J::Float64, g::Float64; χ ::Int= 20, ctmrg_tol::Float64= 1e-10, grad_tol::Float64= 1e-6, maxiter::Int=1000)
    H = transverse_field_ising(PEPSKit.InfiniteSquare(); g)
    peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χ)), peps₀; tol=ctmrg_tol)
    peps, env, E, = fixedpoint(H, peps₀, env₀; tol=grad_tol, boundary_alg=(; tol=ctmrg_tol), 
                              optimizer_alg=(; maxiter=maxiter))
    ξ_h, ξ_v, λ_h, λ_v = correlation_length(peps, env)
    return E, ξ_h, ξ_v, λ_h, λ_v
end

