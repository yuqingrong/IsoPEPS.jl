"""
Physical constants and standard matrices for TFIM calculations.
"""

import numpy as np

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

# Spin states
SPIN_UP = np.array([1, 0], dtype=complex)
SPIN_DOWN = np.array([0, 1], dtype=complex)
