# ---------------------------------------------------------------------------------------------------------------------
# fn_adam_optimizer_stiefel.jl is a SELF-CREATED file implementing an alternative optimizer update step 
# directly on the STIEFEL manifold instead of on a HOMOGENEOUS SPACE
# ---------------------------------------------------------------------------------------------------------------------



#region Section 1: struct StiefelAdamOptimizer
mutable struct StiefelAdamOptimizer{T<:Real} <: OptimizerMethod
    η::T
    ρ₁::T
    ρ₁t::T
    ρ₂::T
    ρ₂t::T
    δ::T

    function StiefelAdamOptimizer(η = 1f-3, ρ₁ = 9f-1, ρ₁t = 9f-1, ρ₂ = 9.9f-1, ρ₂t = 9.9f-1, δ = 3f-7; T=typeof(η)) 
         new{T}(T(η), T(ρ₁), T(ρ₁t), T(ρ₂), T(ρ₂t), T(δ))
    end
end

function StiefelAdamOptimizer(T::Type)
    StiefelAdamOptimizer(T(1f-3))
end

#endregion


#region Section 2: update step

# classic adam update step for non-manifold layer (symplectic preprocessing)
function update!(o::Optimizer{<:StiefelAdamOptimizer{T}}, C::AdamCache, B::AbstractArray, ::AbstractArray) where T
    add!(C.B₁, ((o.method.ρ₁ - o.method.ρ₁t)/(T(1.) - o.method.ρ₁t)) .* C.B₁, ((T(1.) - o.method.ρ₁)/(T(1.) - o.method.ρ₁t)) .* B)
    add!(C.B₂, ((o.method.ρ₂ - o.method.ρ₂t)/(T(1.) - o.method.ρ₂t)) .* C.B₂, ((T(1.) - o.method.ρ₂)/(T(1.) - o.method.ρ₂t)) .* ⊙²(B))
    B .= -o.method.η .* /ᵉˡᵉ(C.B₁, scalar_add(racᵉˡᵉ(C.B₂), o.method.δ))
end

# SELF-CREATED modified "pseudo-adam" update step to perform update on Stiefel manifold directly
function update!(o::Optimizer{<:StiefelAdamOptimizer{T}}, C::AdamCache, B::AbstractArray, Y::StiefelManifold) where T
    # determine B2 from Cache
    C.B₂ .= racᵉˡᵉ( scalar_add(((o.method.ρ₂ - o.method.ρ₂t)/(T(1.) - o.method.ρ₂t)) .* (C.B₁.^2) + ((T(1.) - o.method.ρ₂)/(T(1.) - o.method.ρ₂t)) .* ⊙²(B), o.method.δ ))
    # update Cache:
    add!(C.B₁, ((o.method.ρ₁ - o.method.ρ₁t)/(T(1.) - o.method.ρ₁t)) .* C.B₁, ((T(1.) - o.method.ρ₁)/(T(1.) - o.method.ρ₁t)).* B)
    # get skew symmetric part of Cache divided by B2:
    n = size(C.B₂,2)
    B2n = (C.B₂[1:n,:]' .+ C.B₂[1:n,:]) ./ 2
    W = Y.A'*C.B₁
    # get orthogonal part of Cache divided by B2:
    OB2 = /ᵉˡᵉ(C.B₁ .- Y.A*W, C.B₂)
    # store update vector in gradient varible B
    B .= -o.method.η .* ( Y.A * /ᵉˡᵉ(W,B2n) .+ OB2 .- Y.A*(Y.A'*OB2) )
end

# defaults: 
⊙²(A::AbstractVecOrMat) = A.^2
racᵉˡᵉ(A::AbstractVecOrMat) = sqrt.(A)
/ᵉˡᵉ(A::AbstractVecOrMat, B::AbstractVecOrMat) = A./B
scalar_add(A::AbstractVecOrMat, δ::Real) = A .+ δ

#endregion