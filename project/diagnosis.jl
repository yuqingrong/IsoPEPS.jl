using IsoPEPS
using JSON3
using Yao, YaoBlocks

# Load your result
J=1.0; g = 0.75; row=1; nqubits=3; p=3
data_dir = joinpath(@__DIR__, "results")
datafile = joinpath(data_dir, "circuit_J=1.0_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")

# reconstruct_gates returns (gates, rho, gap, eigenvalues)
gates, rho, gap, eigenvalues = reconstruct_gates(datafile; plot=false)

# virtual_qubits = boundary qubits = (nqubits-1)/2
virtual_qubits = (nqubits - 1) ÷ 2  # = 1 for nqubits=3

# Run diagnosis
diag = diagnose_transfer_channel(gates, row, virtual_qubits)

# Visualize diagnosis
save_path = joinpath(data_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_diagnosis.pdf")
fig = plot_diagnosis(diag; title="Channel Diagnosis: row=$row, g=$g", save_path=save_path)
display(fig)