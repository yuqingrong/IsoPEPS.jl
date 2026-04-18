module PEPSKitExt

using IsoPEPS
using PEPSKit, TensorKit, MPSKitModels
using PEPSKit: InfiniteSquare
using MPSKitModels: transverse_field_ising

function IsoPEPS.pepskit_ground_state(d::Int, D::Int, J::Float64, g::Float64;
                               χ::Int=20, ctmrg_tol::Float64=1e-10,
                               grad_tol::Float64=1e-6, maxiter::Int=1000)
    H = transverse_field_ising(PEPSKit.InfiniteSquare(); g=g)
    peps₀ = InfinitePEPS(ComplexSpace(d), ComplexSpace(D))
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χ)), peps₀; tol=ctmrg_tol)

    peps, env, E, = fixedpoint(H, peps₀, env₀;
                                tol=grad_tol,
                                boundary_alg=(; tol=ctmrg_tol),
                                optimizer_alg=(; maxiter=maxiter))

    ξ_h, ξ_v, λ_h, λ_v = PEPSKit.correlation_length(peps, env)
    ξ = max(maximum(ξ_h), maximum(ξ_v))

    return (energy=E, correlation_length=ξ, ξ_horizontal=ξ_h, ξ_vertical=ξ_v,
            peps=peps, env=env)
end

end # module PEPSKitExt

result = IsoPEPS.pepskit_ground_state(2, 2, 1.0, 2.25; χ=30)
                                                                                                                                                
data = Dict(    
    "energy" => real(result.energy),
    "correlation_length" => result.correlation_length,
    "parameters" => Dict("d"=>2, "D"=>2, "J"=>1.0, "g"=>2.25, "χ"=>20)
)

open("project/results/pepskit_d=2_D=2_g=3.0.json", "w") do io
    JSON3.pretty(io, data)
end