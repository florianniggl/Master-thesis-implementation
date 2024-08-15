# ---------------------------------------------------------------------------------------------------------------------
# fn_single_soliton.jl is a SELF-CREATED file implementing analytical solution and initial/boundary conditions
# of single_soltion example
# ---------------------------------------------------------------------------------------------------------------------



function single_soliton_u₀(ξ::T, ν::T) where T <: Real
    s = sqrt( 1 - ν^2)
    4 * atan( exp( ξ / s ) ) 
end

function single_soliton_u₁(ξ::T, ν::T) where T <: Real
    s = sqrt( 1 - ν^2)
    ( -4 * ν * exp( ξ / s ) )  /  ( s * ( 1 + exp( 2*ξ / s ) ) )
end

function single_soliton_φ(t::T, ν::T, a::T) where T <: Real
    s = sqrt( 1 - ν^2)
    4 * atan( exp( (a - ν*t) / s ) )
end

function single_soliton_ψ(t::T, ν::T, b::T) where T <: Real
    s = sqrt( 1 - ν^2)
    4 * atan( exp( (b - ν*t) / s ) )
end

function single_soliton_exact_sol(t::Union{T,ForwardDiff.Dual{T}}, ξ::T, ν::T) where T <: Real
    s = sqrt( 1 - ν^2)
    4 * atan( exp( (ξ - ν*t) / s ) )
end

