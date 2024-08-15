# ---------------------------------------------------------------------------------------------------------------------
# fn_batch_v2.jl EXTENDS batch.jl from GeometricMachineLearning@v0.2.0 
# ---------------------------------------------------------------------------------------------------------------------


#region Section 1: batch function

# CHANGE fn: added method: 
# to work with AT<:AbstractArray{T, 2} instead of AT<:AbstractArray{T, 3}
function (batch::Batch{<:Nothing})(dl::DataLoader{T, AT}) where {T, AT<:AbstractArray{T, 2}}
    indices = shuffle(1:dl.n_params)
    n_batches = Int(ceil(dl.n_params/batch.batch_size))
    batches = ()
    for batch_number in 1:(n_batches-1)
        batches = (batches..., indices[(batch_number-1)*batch.batch_size + 1:batch_number*batch.batch_size])
    end
 
    # this is needed because the last batch may not have the full size
    batches = (batches..., indices[( (n_batches-1) * batch.batch_size + 1 ):end])
    batches
end

#endregion


#region Section 2: optimize_for_one_epoch!

# CHANGE fn: added method: 
#   - to customize optimize_for_one_epoch! for my purpose: AbstractArray{T, 3} -> AbstractArray{T, 2}
#   - removed output_batch as unobserved learning
#   - added argument progress_object
function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT}, batch::Batch, loss, Lo::Loss, progress_object) where {T, AT<:AbstractArray{T, 2}}
    count = 0
    total_error = T(0)
    batches = batch(dl)
    @views for batch_indices in batches 
        count += 1
        # these `copy`s should not be necessary! coming from a Zygote problem!
        input_batch = copy(dl.input[:, batch_indices])
        type = typeof(input_batch)
        loss_value, pullback = Zygote.pullback(ps -> loss(model, ps, input_batch, Lo), ps)
        total_error += loss_value
        dp = pullback(one(loss_value))[1]
        optimization_step!(opt, model, ps, dp)
        ProgressMeter.next!(progress_object; showvalues=[(:TrainingLoss, loss_value)])
    end
    total_error/count
end

# CHANGE fn: added method: 
#   - to customize optimize_for_one_epoch! for my purpose: AbstractArray{T, 3} -> AbstractArray{T, 2}
#   - removed output_batch as unobserved learning
#   - added argument progress_object
#   - to work with alternative_optimizer: distinguish different metrics and vector transports
#   - to work wit differnt loss functions: added loss function type argument Lo
function optimize_for_one_epoch!(opt::Optimizer, met::Metric, vt::Vectortransport, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT}, batch::Batch, loss, Lo::Loss, progress_object) where {T, AT<:AbstractArray{T, 2}}
    count = 0
    total_error = T(0)
    batches = batch(dl)
    @views for batch_indices in batches 
        count += 1
        # these `copy`s should not be necessary! coming from a Zygote problem!
        input_batch = copy(dl.input[:, batch_indices])
        type = typeof(input_batch)
        loss_value, pullback = Zygote.pullback(ps -> loss(model, ps, input_batch, Lo), ps)
        total_error += loss_value
        dp = pullback(one(loss_value))[1]
        optimization_step!(opt, met, vt, model, ps, dp)
        ProgressMeter.next!(progress_object; showvalues=[(:TrainingLoss, loss_value)])
    end
    total_error/count
end

# CHANGE fn: added method: 
# to work with AbstractArray{T, 2}
function optimize_for_one_epoch!(opt::Optimizer, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT}, batch::Batch, Lo::Loss, progress_object) where {T, AT<:AbstractArray{T, 2}}
    optimize_for_one_epoch!(opt, model, ps, dl, batch, loss, Lo, progress_object)
end

# CHANGE fn: added method: 
#   - to work with AbstractArray{T, 2}
#   - to work with alternative_optimizer: distinguish different metrics and vector transports
function optimize_for_one_epoch!(opt::Optimizer, met::Metric, vt::Vectortransport, model, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT}, batch::Batch, Lo::Loss, progress_object) where {T, AT<:AbstractArray{T, 2}}
    optimize_for_one_epoch!(opt, met, vt, model, ps, dl, batch, loss, Lo, progress_object)
end

#endregion