# ---------------------------------------------------------------------------------------------------------------------
# fn: fn_init_optimizer_cache.jl MODIFIES init_optimizer_cache.jl from GeometricMachineLearning@v0.2.0 
# ---------------------------------------------------------------------------------------------------------------------



init_optimizer_cache(::GradientOptimizer, x) = setup_gradient_cache(x)
init_optimizer_cache(::MomentumOptimizer, x) = setup_momentum_cache(x)
# fn MODIFIED: added custom (adam) optimizer types
init_optimizer_cache(::Union{AdamOptimizer, StiefelAdamOptimizer, StiefelAdamOptimizerWithDecay, StiefelStochasticGradientDescentWithMomentum, StiefelStochasticGradientDescentWithMomentumWithDecay, StiefelCayleyAdamOptimizerFromOtherPaper}, x) = setup_adam_cache(x)
init_optimizer_cache(::BFGSOptimizer, x) = setup_bfgs_cache(x)

setup_adam_cache(ps::Tuple) = Tuple([setup_adam_cache(x) for x in ps])
setup_adam_cache(ps::NamedTuple) = apply_toNT(setup_adam_cache, ps)
setup_adam_cache(B::AbstractArray{<:Number}) = AdamCache(B)

# fn MODIFIED: overriting GeometricMachineLearning@v0.2.0's method to initialize AdamCache with default Base.zeros 
# instead of initializing as StiefelLieHorAlgMatrix (original implementation is left as comment)
function Base.zero(Y::StiefelManifold{T}) where T 
    zero(Y.A)
    #N, n = size(Y)
    #backend = KernelAbstractions.get_backend(Y.A)
    #zeros(backend, StiefelLieAlgHorMatrix{T}, N, n)
    #zeros(backend, AbstractArray{T}, N, n)
end