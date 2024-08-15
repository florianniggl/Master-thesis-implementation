# ---------------------------------------------------------------------------------------------------------------------
# fn_stochastic_gradient_descent_with_momentum_withDecay.jl is a SELF-CREATED file implementing an alternative optimizer 
# update step directly on the STIEFEL manifold instead of on a HOMOGENEOUS SPACE
# ---------------------------------------------------------------------------------------------------------------------



#region Section 1: struct StiefelStochasticGradientDescentWithMomentumWithDecay

mutable struct StiefelStochasticGradientDescentWithMomentumWithDecay{T<:Real} <: OptimizerMethod
    η::T
    α::T

    StiefelStochasticGradientDescentWithMomentumWithDecay(η = 1f-3, α = 9f-1; T=typeof(η)) = new{T}(T(η), T(α))
end

function StiefelStochasticGradientDescentWithMomentumWithDecay(T::Type)
    StiefelStochasticGradientDescentWithMomentumWithDecay(T(1f-3))
end

#endregion


#region Section 2: update step

# SELF-CREATED SGD with Decay update step to perform update on Stiefel manifold directly (works both for manifold and 
# non manifold layer).
function update!(o::Optimizer{<:StiefelStochasticGradientDescentWithMomentumWithDecay{T}}, C::AdamCache, B::AbstractArray, ::AbstractArray) where T
    add!(C.B₁, o.method.α .* C.B₁, - o.method.η .* B)
    B .= C.B₁
end

# defaults: 
⊙²(A::AbstractVecOrMat) = A.^2
racᵉˡᵉ(A::AbstractVecOrMat) = sqrt.(A)
/ᵉˡᵉ(A::AbstractVecOrMat, B::AbstractVecOrMat) = A./B
scalar_add(A::AbstractVecOrMat, δ::Real) = A .+ δ

#endregion