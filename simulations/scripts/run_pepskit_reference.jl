"""
PEPSKit reference computation runner.

Wraps the existing reference.jl logic for use from simulations/.
Run from the simulations/ directory:
    julia --project=simulations simulations/scripts/run_pepskit_reference.jl
"""

include(joinpath(@__DIR__, "..", "..", "project", "reference.jl"))
