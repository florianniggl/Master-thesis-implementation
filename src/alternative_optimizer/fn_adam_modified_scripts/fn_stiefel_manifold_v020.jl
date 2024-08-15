# ---------------------------------------------------------------------------------------------------------------------
# fn_stiefel_manifold_v2.jl EXTENDS stiefel_manifold.jl from GeometricMachineLearning@v0.2.0
# ---------------------------------------------------------------------------------------------------------------------



#fn_ADDED: Euclidean metric rgrad
function rgrad(::Euclidean, Y::StiefelManifold, e_grad::AbstractMatrix)
    H = Y.A'*e_grad
    e_grad .- (Y.A ./ 2) * (H .+ H')
end

# Canonical metric rgrad
function rgrad(::Canonical, Y::StiefelManifold, e_grad::AbstractMatrix)
    e_grad .- Y.A * (e_grad' * Y.A)
end

