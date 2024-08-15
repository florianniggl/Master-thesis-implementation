# ---------------------------------------------------------------------------------------------------------------------
# fn_soliton_soliton_doublets.jl is a SELF-CREATED file implementing analytical solution and initial/boundary conditions
# of soliton_soliton_doublets example
# ---------------------------------------------------------------------------------------------------------------------



function soliton_soliton_doublets_u₀(ξ::T, ν::T) where T <: Real
    s = sqrt(1 - ν^2)
    4 * atan( ν * sinh(ξ / s) )
end

function soliton_soliton_doublets_u₁(ξ::T, ν::T) where T <: Real
    0
end

function soliton_soliton_doublets_φ(t::T, ν::T, a::T) where T <: Real
    s = sqrt(1 - ν^2)
    4 * atan( ν * sech(ν*t / s) * sinh(a / s) )
end

function soliton_soliton_doublets_ψ(t::T, ν::T, b::T) where T <: Real
    s = sqrt(1 - ν^2)
    4 * atan( ν * sech(ν*t / s) * sinh(b / s) )
end

function soliton_soliton_doublets_exact_sol(t::Union{T,ForwardDiff.Dual{T}}, ξ::T, ν::T) where T <: Real
    s = sqrt(1 - ν^2)
    4 * atan( ν * sech(ν*t / s) * sinh(ξ / s) )
end
