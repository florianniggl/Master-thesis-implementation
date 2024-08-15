# ---------------------------------------------------------------------------------------------------------------------
# fn_retraction.jl EXTENDS retraction.jl from GeometricMachineLearning@v0.2.0
# ---------------------------------------------------------------------------------------------------------------------



#region Section 1: retraction!

#fn_ADDED: to work with cayley directly on Stiefel tangent space
function retraction!(A::AbstractExplicitLayer, Z::NamedTuple, ps::NamedTuple, C::NamedTuple, ::Any)
    euclideanupdate!(Z, ps) 
end

function retraction!(A::LayerWithManifold{M, N, Cayley}, Z::NamedTuple, ps::NamedTuple, C::NamedTuple, vt::Vectortransport) where {M,N}
    cayley!(Z, ps, C, vt)
end

#endregion


#region Section 2: Cayley on Stiefel

# to work with cayley on Stiefel directly: implements custom layer wise application 
cayley!(Z::NamedTuple, ps::NamedTuple, C::NamedTuple, vt::Vectortransport) = apply_toNT(cayley!, vt, Z, ps, C)

# to work with cayley on Stiefel directly: implements vector transport SUBMANIFOLD
function cayley!(::SubmanifoldVectortransport, Z::AbstractArray{T}, Φ::AbstractArray{T}, C::AdamCache) where T
    H = Φ.A'*Z
    U = hcat(Z .- (Φ.A ./ 2)*(H .- H'), -Φ.A)
    V = vcat(Φ.A',Z')
    VU = V*U
    VΦ = V*Φ

    # apply retraction to update the weights
    Φ.A .= Φ.A .+ (U ./ 2) * ( VΦ .+ (one(VU) .- (VU ./ 2)) \ (VΦ .+ (VU ./ 2)*VΦ) )

    # vector transport *sub* of Cache C.B₁ along Cayley 
    H .= Φ.A'*C.B₁
    C.B₁ .= C.B₁ .- (Φ.A ./ 2) * ( H .+ H' )
end

# to work with cayley on Stiefel directly: implements vector transport DIFFERENTIAL
function cayley!(::DifferentialVectortransport, Z::AbstractArray{T}, Φ::AbstractArray{T}, C::AdamCache) where T
    H = Φ.A'*Z
    U = hcat(Z .- (Φ.A ./ 2)*(H .- H'), -Φ.A)
    V = vcat(Φ.A',Z')
    VU = V*U
    VΦ = V*Φ

    # vector transport *diff* of Cache C.B₁ along Cayley 
    H .= Φ.A'*C.B₁
    U_B = hcat(C.B₁ .- (Φ.A ./ 2)*(H .- H'), -Φ.A)
    V_B = vcat(Φ.A',C.B₁')
    VU_B_half = (V*U_B) ./ 2
    mat_to_be_inv = one(VU) .- (VU ./ 2)
    C.B₁ .= (U_B .+ U * (mat_to_be_inv \ VU_B_half)) * (V_B * (Φ.A .+ U * (mat_to_be_inv \ (VΦ ./ 2))))

    # apply retraction to update the weights
    Φ.A .= Φ.A .+ (U ./ 2) * ( VΦ .+ (one(VU) .- (VU ./ 2)) \ (VΦ .+ (VU ./ 2)*VΦ) )
end

#endregion


#region Section 3: Euclidean update

#fn_ADDED: to work with symplectic layers in euclidean update step
euclideanupdate!(Z::NamedTuple, ps::NamedTuple) = apply_toNT(euclideanupdate!, Z, ps)
function euclideanupdate!(Z::AbstractArray{T}, Y::AbstractArray) where T 
    Y .= Y .+ Z 
end

#endregion
