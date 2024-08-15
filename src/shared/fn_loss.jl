# ---------------------------------------------------------------------------------------------------------------------
# fn_loss.jl MODIFIES data_loader.jl from GeometricMachineLearning@v0.2.0 
# ---------------------------------------------------------------------------------------------------------------------


# ADDED fn: original loss method for autoencoder in GML (from file data_loader.jl)
# needed to be available in this project for self implemented update step routine in training.jl
function loss(model::Chain, ps::Tuple, input::BT, ::LossGML) where {T, BT<:AbstractArray{T}} 
    output_estimate = model(input, ps)
    norm(output_estimate - input) / norm(input) # /T(sqrt(size(input, 2)*size(input, 3)))
end

# ADDED fn: loss function for autoencoder as proposed by paper "Symplectic Model Reduction of Hamiltonian 
# Systems on Nonlinear manifolds
function loss(model::Chain, ps::Tuple, input::BT, ::LossModified) where {T, BT<:AbstractArray{T}} 
    output_estimate = model(input, ps)
    N,k = size(input)
    1/(N*k)*norm(output_estimate - input)^2
    #(1/N*norm(input))*(norm(output_estimate - input)^2)
end