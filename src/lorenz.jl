"""
    Point{D, T}

A point in D-dimensional space, with coordinates of type T.

# Examples
```jldoctest
julia> p1 = Point(1.0, 2.0)
Point{2, Float64}((1.0, 2.0))

julia> p2 = Point(3.0, 4.0)
Point{2, Float64}((3.0, 4.0))

julia> p1 + p2
Point{2, Float64}((4.0, 6.0))
```
"""
# define a point in D-dimensional space
struct Point{D, T <: Real}
    data::NTuple{D, T}  # a tuple of D elements of type T
end
Point(x::Real...) = Point((x...,))  # `...` is the splat operator
# define 2D and 3D points
const Point2D{T} = Point{2, T}
const Point3D{T} = Point{3, T}

# define the dot product of two coordinate vectors
# `mapreduce` is a high-order function that applies a function to each element 
# of an iterable and then reduces the result to a single value.
LinearAlgebra.dot(x::Point, y::Point) = mapreduce(*, +, x.data, y.data)
# implement the operations of the point
# `Base` is the standard library of Julia
# `Base.isapprox` is used to define a new method for the function `isapprox` in the `Base` module
# for arithmetic operations like `*`, `*`, `+`, an extra `:` is required to avoid ambiguity
Base.:*(x::Real, y::Point) = Point(x .* y.data) # `.` is the broadcast operator
Base.:/(y::Point, x::Real) = Point(y.data ./ x)
Base.:+(x::Point, y::Point) = Point(x.data .+ y.data)
Base.isapprox(x::Point, y::Point; kwargs...) = all(isapprox.(x.data, y.data; kwargs...))
# `all` returns true if all elements of the iterable are true

# define the index and broadcastable functions
Base.getindex(p::Point, i::Int) = p.data[i] # for `p[i]` like operations
Base.broadcastable(p::Point) = p.data # for `x .+ y` like operations
Base.iterate(p::Point, args...) = iterate(p.data, args...) # for `[p...]` like operations

# the Lorenz system
struct Lorenz
    σ::Float64
    ρ::Float64
    β::Float64
end

# the differential equation of the Lorenz system
function field(p::Lorenz, u)
    x, y, z = u
    Point(p.σ*(y-x), x*(p.ρ-z)-y, x*y-p.β*z)
end

# abstract type for integrators, which allows us to switch between different integration methods
abstract type AbstractIntegrator end
# Runge-Kutta 4th order method
# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
struct RungeKutta{K} <: AbstractIntegrator end
# Euclidean method
struct Euclidean <: AbstractIntegrator end

function integrate_step(f, ::RungeKutta{4}, t, y, Δt)
    k1 = Δt * f(t, y)
    k2 = Δt * f(t+Δt/2, y + k1 / 2)
    k3 = Δt * f(t+Δt/2, y + k2 / 2)
    k4 = Δt * f(t+Δt, y + k3)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end

# Euclidean integration
function integrate_step(f, ::Euclidean, t, y, Δt)
    return y + Δt * f(t, y)
end

function integrate_step(lz::Lorenz, int::AbstractIntegrator, u, Δt)
    return integrate_step((t, u) -> field(lz, u), int, zero(Δt), u, Δt)
end