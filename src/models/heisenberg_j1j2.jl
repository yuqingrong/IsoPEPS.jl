# =============================================================================
# Heisenberg J1-J2 Model
# =============================================================================

"""
    HeisenbergJ1J2(J1, J2)
    HeisenbergJ1J2(; J1=1.0, J2=0.0)

Heisenberg J1-J2 model: H = J1 Σ_{⟨i,j⟩} Sᵢ·Sⱼ + J2 Σ_{⟨⟨i,j⟩⟩} Sᵢ·Sⱼ
"""
struct HeisenbergJ1J2 <: AbstractModel
    J1::Float64
    J2::Float64
end
HeisenbergJ1J2(; J1=1.0, J2=0.0) = HeisenbergJ1J2(J1, J2)

model_name(::HeisenbergJ1J2) = "heisenberg_j1j2"
needs_y_measurement(::HeisenbergJ1J2) = true
default_unit_cell(::HeisenbergJ1J2) = :two_by_two
model_label(m::HeisenbergJ1J2) = "Heisenberg J1=$(m.J1) J2=$(m.J2)"

function compute_energy_from_samples(m::HeisenbergJ1J2, X_samples, Z_samples, Y_samples, row)
    compute_heisenberg_energy(X_samples, Z_samples, Y_samples, m.J1, m.J2, row)
end

function compute_exact_energy_from_gates(m::HeisenbergJ1J2, gates, row, virtual_qubits;
                                          unit_cell=:single, gates_even=nothing,
                                          optimizer=GreedyMethod())
    if unit_cell == :two_by_two && gates_even !== nothing
        energy = compute_exact_heisenberg_energy_2x2(gates, gates_even, row, virtual_qubits, m.J1, m.J2;
                                                      optimizer=optimizer)
    else
        energy = compute_exact_heisenberg_energy(gates, row, virtual_qubits, m.J1, m.J2;
                                                  optimizer=optimizer)
    end
    return energy, 0.0, 0.0, 0.0
end
