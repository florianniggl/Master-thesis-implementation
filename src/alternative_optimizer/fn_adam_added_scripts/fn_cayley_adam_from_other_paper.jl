# ---------------------------------------------------------------------------------------------------------------------
# fn_cayley_adam_from_other_paper.jl is a SELF-CREATED file implementing an alternative optimizer update step 
# directly on the STIEFEL manifold instead of on a HOMOGENEOUS SPACE.
# This CayleyAdamOptimizer implementation is taken from the paper 
# "Efficient Riemannian Optimization on the Stiefel manifold via the Cayley retraction"
# ---------------------------------------------------------------------------------------------------------------------



#region Section 1: struct StiefelCayleyAdamOptimizerFromOtherPaper

mutable struct StiefelCayleyAdamOptimizerFromOtherPaper{T<:Real} <: OptimizerMethod
    η::T
    ρ₁::T
    ρ₁t::T
    ρ₂::T
    ρ₂t::T
    δ::T
    v₁::T
    q::T
    s::Int

    function StiefelCayleyAdamOptimizerFromOtherPaper(η = 1f-3, ρ₁ = 9f-1, ρ₁t = 9f-1, ρ₂ = 9.9f-1, ρ₂t = 9.9f-1, δ = 3f-8, v₁ = 1.0, q = 0.5, s = 2; T=typeof(η)) 
         new{T}(T(η), T(ρ₁), T(ρ₁t), T(ρ₂), T(ρ₂t), T(δ), T(v₁), T(q), s)
    end
end

function StiefelCayleyAdamOptimizerFromOtherPaper(T::Type)
    StiefelCayleyAdamOptimizerFromOtherPaper(T(1f-3))
end

#endregion


#region Section 2: update step

 # classic adam update step for non-manifold layer (symplectic preprocessing)
function update!(o::Optimizer{<:StiefelCayleyAdamOptimizerFromOtherPaper{T}}, C::AdamCache, B::AbstractArray, ::AbstractArray) where T
    add!(C.B₁, ((o.method.ρ₁ - o.method.ρ₁t)/(T(1.) - o.method.ρ₁t)) .* C.B₁, ((T(1.) - o.method.ρ₁)/(T(1.) - o.method.ρ₁t)).* B)
    add!(C.B₂, ((o.method.ρ₂ - o.method.ρ₂t)/(T(1.) - o.method.ρ₂t)) .* C.B₂, ((T(1.) - o.method.ρ₂)/(T(1.) - o.method.ρ₂t)) .* ⊙²(B))
    B .= B -o.method.η .* /ᵉˡᵉ(C.B₁, scalar_add(racᵉˡᵉ(C.B₂), o.method.δ))
end

# "Cayley ADAM" update step to perform update on Stiefel manifold directly taken from paper "EFFICIENT RIEMANNIAN OPTIMIZATION 
# ON THE STIEFEL MANIFOLD VIA THE CAYLEY TRANSFORM"
function update!(o::Optimizer{<:StiefelCayleyAdamOptimizerFromOtherPaper{T}}, C::AdamCache, B::AbstractArray, Y::StiefelManifold) where T
    C.B₁ .=  (o.method.ρ₁ .* C.B₁) .+ ((1-o.method.ρ₁) .* B)
    o.method.v₁ = o.method.ρ₂*o.method.v₁ + (1-o.method.ρ₂)*norm(B)^2
    v = o.method.v₁ / (1-o.method.ρ₂t)
    r = (1-o.method.ρ₁t)*sqrt(v + o.method.δ) 
    MA = C.B₁*Y.A'
    WH = MA .- (Y.A ./ 2) * (Y.A'*MA)   
    W = (WH .- WH') ./ r
    C.B₁ .= (r .* W) * Y.A
    a = min(o.method.η, 2*o.method.q / (norm(W) + o.method.δ))
    X = Y.A .- (a .* C.B₁)
    for i in 1:o.method.s
        X .= Y.A .- ((a/2) .* W) *(Y.A .+ X)
    end
    Y.A .= X
end

# defaults: 
⊙²(A::AbstractVecOrMat) = A.^2
racᵉˡᵉ(A::AbstractVecOrMat) = sqrt.(A)
/ᵉˡᵉ(A::AbstractVecOrMat, B::AbstractVecOrMat) = A./B
scalar_add(A::AbstractVecOrMat, δ::Real) = A .+ δ

#endregion