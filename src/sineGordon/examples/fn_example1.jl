# ---------------------------------------------------------------------------------------------------------------------
# fn_example1.jl is a SELF-CREATED file implementing analytical solution and initial/boundary conditions of example1
# ---------------------------------------------------------------------------------------------------------------------



function example1_u₀(ξ::T, ν::T) where T <: Real
    0
end

function example1_u₁(ξ::T, ν::T) where T <: Real
    4 * sech(ξ)
end

function example1_φ(t::T, ν::T, a::T) where T <: Real
    4 * atan(t * sech(a))
end

function example1_ψ(t::T, ν::T, b::T) where T <: Real
    4 * atan(t * sech(b))
end

function example1_exact_sol(t::Union{T,ForwardDiff.Dual{T}}, ξ::T, ν::T) where T <: Real
    4 * atan(t * sech(ξ))
end
