# =============================================================================
# Transverse Field Ising Model
# =============================================================================

"""
    TFIM(J, g)
    TFIM(; J=1.0, g=1.0)

Transverse Field Ising Model: H = -g Σ Xᵢ - J Σ ZᵢZⱼ
"""
struct TFIM <: AbstractModel
    J::Float64
    g::Float64
end
TFIM(; J=1.0, g=1.0) = TFIM(J, g)

model_name(::TFIM) = "tfim"
needs_y_measurement(::TFIM) = false
model_label(m::TFIM) = "TFIM J=$(m.J) g=$(m.g)"

function compute_energy_from_samples(m::TFIM, X_samples, Z_samples, ::Any, row)
    compute_tfim_energy(X_samples, Z_samples, m.g, m.J, row)
end

function compute_exact_energy_from_gates(m::TFIM, gates, row, virtual_qubits;
                                          unit_cell=:single, gates_even=nothing,
                                          optimizer=GreedyMethod())
    X_cost = real(compute_X_expectation(nothing, gates, row, virtual_qubits; optimizer=optimizer))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(nothing, gates, row, virtual_qubits; optimizer=optimizer)
    ZZ_vert = real(ZZ_vert)
    ZZ_horiz = real(ZZ_horiz)
    energy = -m.g * X_cost - m.J * (row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
    return energy, X_cost, ZZ_vert, ZZ_horiz
end
