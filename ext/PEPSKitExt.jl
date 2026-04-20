module PEPSKitExt

using IsoPEPS
using PEPSKit, TensorKit, MPSKitModels
using PEPSKit: InfiniteSquare
using MPSKitModels: transverse_field_ising
using Logging
using JSON3
struct GaugeFilterLogger{L<:AbstractLogger} <: AbstractLogger
    inner::L
end
function Logging.handle_message(l::GaugeFilterLogger, level, message, _module, group, id, file, line; kwargs...)
    msg = string(message)
    if occursin("cotangents sensitive to gauge", msg) ||
       occursin("cotangent linear problem", msg) ||
       occursin("cotangent problem did not converge", msg) ||
       occursin("Fixed-point gradient computation using Arnoldi failed", msg) ||
       occursin("Falling back to linear solver", msg) ||
       occursin("Arnoldi eigsolve stopped without convergence", msg) ||
       occursin("Linesearch not converged", msg) ||
       occursin("Linesearch bracket converged", msg)
        return nothing
    end
    Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
end
Logging.shouldlog(l::GaugeFilterLogger, level, _module, group, id) = Logging.shouldlog(l.inner, level, _module, group, id)
Logging.min_enabled_level(l::GaugeFilterLogger) = Logging.min_enabled_level(l.inner)
Logging.catch_exceptions(l::GaugeFilterLogger) = Logging.catch_exceptions(l.inner)

function IsoPEPS.pepskit_ground_state(d::Int, D::Int, J::Float64, g::Float64;
                               χ::Int=20, ctmrg_tol::Float64=1e-8,
                               grad_tol::Float64=1e-4, maxiter::Int=100,
                               peps_init=nothing, env_init=nothing)
    H = transverse_field_ising(PEPSKit.InfiniteSquare(); g=g)
    peps₀ = isnothing(peps_init) ?
        InfinitePEPS(randn, Float64, ℝ^d, ℝ^D) :
        peps_init
    env₀ = isnothing(env_init) ?
        first(leading_boundary(CTMRGEnv(peps₀, ℝ^χ), peps₀; tol=ctmrg_tol)) :
        env_init

    filtered = GaugeFilterLogger(current_logger())

    peps, env, E = with_logger(filtered) do
        peps, env, E, = fixedpoint(H, peps₀, env₀;
                                    tol=grad_tol,
                                    boundary_alg=(; tol=ctmrg_tol),
                                    gradient_alg=(; iterscheme=:diffgauge),
                                    optimizer_alg=(; maxiter=maxiter))
        peps, env, E
    end

    ξ_h, ξ_v, λ_h, λ_v = PEPSKit.correlation_length(peps, env)
    ξ = ξ_h

    return (energy=E, correlation_length=ξ, ξ_horizontal=ξ_h, ξ_vertical=ξ_v,
            peps=peps, env=env)
end

end # module PEPSKitExt