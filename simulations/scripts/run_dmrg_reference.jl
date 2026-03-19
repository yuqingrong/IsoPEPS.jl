"""
DMRG reference computation runner.

Wraps the existing dmrg_reference.jl logic for use from simulations/.
Run from the simulations/ directory:
    julia --project=simulations simulations/scripts/run_dmrg_reference.jl
"""

# The DMRG reference code lives in project/dmrg_reference.jl
# This script provides a clean entry point from the simulations/ directory.
# For now, include the original script directly.

include(joinpath(@__DIR__, "..", "..", "project", "dmrg_reference.jl"))
