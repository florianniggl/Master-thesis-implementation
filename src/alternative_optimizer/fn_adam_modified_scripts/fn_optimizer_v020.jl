# ---------------------------------------------------------------------------------------------------------------------
# fn_optimizer.jl MODIFIES and EXTENDS optimizer.jl from GeometricMachineLearning@v0.2.0 to work WITHOUT
# homogeneous space
# ---------------------------------------------------------------------------------------------------------------------



#region Section 1: custom Optimizer constructor

# fn_MODIFIED: custom Optimizer constructor to work with modified fn_optmizer_init_cache.jl file
function Optimizer(m::OptimizerMethod, x)
    Optimizer(m, init_optimizer_cache(m, x), 0)
end

#endregion


#region Section 2: custom layerwise optimization_step

# fn_ADDED: custom optimization step to work without homogeneous space
function optimization_step!(o::Optimizer, met::Metric, vt::Vectortransport, d::Union{AbstractExplicitLayer, AbstractExplicitCell}, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx, met)
    update!(o, C, gx, ps)
    retraction!(d, gx, ps, C, vt)
end

# fn_ADDED: special optimization_step! for CayleyADAM from other paper as comparison optimizer
function optimization_step!(o::Optimizer{StiefelCayleyAdamOptimizerFromOtherPaper{T}, CT}, met::Metric, vt::Vectortransport, d::Union{AbstractExplicitLayer, AbstractExplicitCell}, ps::NamedTuple, C::NamedTuple, dx::NamedTuple) where {T, CT}
    update!(o, C, dx, ps)
end

#endregion


#region Section 3: custom network optimization_step (depending on choice of custom optimizer)

# fn_ADDED: to work with StiefelAdamOptimizer
function optimization_step!(o::Optimizer{StiefelAdamOptimizer{T}, CT}, met::Metric, vt::Vectortransport, model::Chain, ps::Tuple, dx::Tuple) where {T, CT}
    o.step += 1
    # fn_ADDED: to update exponent parameters efficiently
    o.method.ρ₁t *= o.method.ρ₁  
    o.method.ρ₂t *= o.method.ρ₂
    for (index, element) in zip(eachindex(model.layers), model.layers)
        optimization_step!(o, met, vt, element, ps[index], o.cache[index], dx[index])
    end
end

# fn_ADDED: to work with StiefelAdamOptimizerWithDecay
function optimization_step!(o::Optimizer{StiefelAdamOptimizerWithDecay{T}, CT}, met::Metric, vt::Vectortransport, model::Chain, ps::Tuple, dx::Tuple) where {T, CT}
    o.step += 1
    # fn_ADDED: to update exponent parameters efficiently
    o.method.ρ₁t *= o.method.ρ₁  
    o.method.ρ₂t *= o.method.ρ₂
    o.method.η *= 0.9995
    for (index, element) in zip(eachindex(model.layers), model.layers)
        optimization_step!(o, met, vt, element, ps[index], o.cache[index], dx[index])
    end
end

# fn_ADDED: to work with StiefelStochasticGradientDescentWithMomentum
function optimization_step!(o::Optimizer{StiefelStochasticGradientDescentWithMomentum{T}, CT}, met::Metric, vt::Vectortransport, model::Chain, ps::Tuple, dx::Tuple) where {T, CT}
    o.step += 1
    for (index, element) in zip(eachindex(model.layers), model.layers)
        optimization_step!(o, met, vt, element, ps[index], o.cache[index], dx[index])
    end
end

# fn_ADDED: to work with StiefelStochasticGradientDescentWithMomentumWithDecay
function optimization_step!(o::Optimizer{StiefelStochasticGradientDescentWithMomentumWithDecay{T}, CT}, met::Metric, vt::Vectortransport, model::Chain, ps::Tuple, dx::Tuple) where {T, CT}
    o.step += 1
    o.method.η *= 0.9995
    o.method.α *= 0.9995
    for (index, element) in zip(eachindex(model.layers), model.layers)
        optimization_step!(o, met, vt, element, ps[index], o.cache[index], dx[index])
    end
end

# fn_ADDED: to work with StiefelCayleyAdamOptimizerFromOtherPaper
function optimization_step!(o::Optimizer{StiefelCayleyAdamOptimizerFromOtherPaper{T}, CT}, met::Metric, vt::Vectortransport, model::Chain, ps::Tuple, dx::Tuple) where {T, CT}
    o.step += 1
    o.method.ρ₁t *= o.method.ρ₁  
    o.method.ρ₂t *= o.method.ρ₂
    o.method.η *= 0.9995
    for (index, element) in zip(eachindex(model.layers), model.layers)
        optimization_step!(o, met, vt, element, ps[index], o.cache[index], dx[index])
    end
end

#endregion


#region Section 4: util functions

rgrad(ps::NamedTuple, dx::NamedTuple, met::Any) = apply_toNT(rgrad, met, ps, dx)
#rgrad(ps::NamedTuple, dx::NamedTuple, met::Any) = apply_toNT(rgrad, ps, dx)

function rgrad(::Any, Y::AbstractVecOrMat, dx::AbstractVecOrMat)
#function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end

# fn_MODIFIED: to work with changed update without homogeneous space!
function update!(m::Optimizer, C::NamedTuple, B::NamedTuple, ps::NamedTuple)
    apply_toNT(m, C, B, ps, update!)
end

function apply_toNT(m::Optimizer, ps₁::NamedTuple, ps₂::NamedTuple, ps₃::NamedTuple, fun_name)    
    apply_toNT((ps₁, ps₂, ps₃) -> fun_name(m, ps₁, ps₂, ps₃), ps₁, ps₂, ps₃)
end

#endregion