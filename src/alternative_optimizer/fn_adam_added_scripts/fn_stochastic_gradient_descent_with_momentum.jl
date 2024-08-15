# ---------------------------------------------------------------------------------------------------------------------
# fn_stochastic_gradient_descent_with_momentum.jl is a SELF-CREATED file implementing an alternative optimizer 
# update step directly on the STIEFEL manifold instead of on a HOMOGENEOUS SPACE
# ---------------------------------------------------------------------------------------------------------------------



#region Section 1: struct StiefelStochasticGradientDescentWithMomentum

mutable struct StiefelStochasticGradientDescentWithMomentum{T<:Real} <: OptimizerMethod
    η::T
    α::T

    function StiefelStochasticGradientDescentWithMomentum(η = 1f-3, α = 9f-1; T=typeof(η))
        new{T}(T(η), T(α))
    end
end

function StiefelStochasticGradientDescentWithMomentum(T::Type)
    StiefelStochasticGradientDescentWithMomentum(T(1f-3))
end

#endregion


#region Section 2: update step

# SELF-CREATED SGD update step to perform update on Stiefel manifold directly (works both for manifold and 
# non manifold layer).
function update!(o::Optimizer{<:StiefelStochasticGradientDescentWithMomentum{T}}, C::AdamCache, B::AbstractArray, ::AbstractArray) where T
    add!(C.B₁, o.method.α .* C.B₁, - o.method.η .* B)
    B .= C.B₁
end

# defaults: 
⊙²(A::AbstractVecOrMat) = A.^2
racᵉˡᵉ(A::AbstractVecOrMat) = sqrt.(A)
/ᵉˡᵉ(A::AbstractVecOrMat, B::AbstractVecOrMat) = A./B
scalar_add(A::AbstractVecOrMat, δ::Real) = A .+ δ

#endregion