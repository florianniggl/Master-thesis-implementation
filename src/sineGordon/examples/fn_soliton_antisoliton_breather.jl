# ---------------------------------------------------------------------------------------------------------------------
# fn_soliton_antisoliton_breather.jl is a SELF-CREATED file implementing analytical solution and initial/boundary conditions
# of soliton_antisoliton_breather example
# ---------------------------------------------------------------------------------------------------------------------



function soliton_antisoliton_breather_u₀(ξ::T, ν::T) where T <: Real
    0
end

function soliton_antisoliton_breather_u₁(ξ::T, ν::T) where T <: Real
    s = sqrt(1 + ν^2)
    (4 / s) * sech(ξ / s) 
end

function soliton_antisoliton_breather_φ(t::T, ν::T, a::T) where T <: Real
    s = sqrt(1 + ν^2)
    4 * atan( (1 / ν) * sin(ν*t / s) * sin(a / s) )
end

function soliton_antisoliton_breather_ψ(t::T, ν::T, b::T) where T <: Real
    s = sqrt(1 + ν^2)
    4 * atan( (1 / ν) * sin(ν*t / s) * sin(b / s) )
end

function soliton_antisoliton_breather_exact_sol(t::Union{T,ForwardDiff.Dual{T}}, ξ::T, ν::T) where T <: Real
    s = sqrt(1 + ν^2)
    4 * atan( (1 / ν) * sin(ν*t / s) * sin(ξ / s) )
end

