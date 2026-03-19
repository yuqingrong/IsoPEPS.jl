# =============================================================================
# Abstract Model Interface
# =============================================================================

"""
    AbstractModel

Abstract supertype for all physical models.

Each concrete model must implement:
- `compute_energy_from_samples(model, X, Z, Y, row)` → Float64
- `model_name(model)` → String

Optional overrides:
- `needs_y_measurement(model)` → Bool (default: false)
- `default_unit_cell(model)` → Symbol (default: :single)
- `model_label(model)` → String (for logging)
"""
abstract type AbstractModel end

"""Whether this model requires Y-basis measurements."""
needs_y_measurement(::AbstractModel) = false

"""Default unit cell type for this model."""
default_unit_cell(::AbstractModel) = :single

"""Human-readable label for logging."""
model_label(m::AbstractModel) = model_name(m)

"""String identifier for the model (used in filenames, dispatch keys)."""
function model_name end

"""Compute energy from measurement samples."""
function compute_energy_from_samples end

"""Compute exact energy from gates via tensor contraction."""
function compute_exact_energy_from_gates end
