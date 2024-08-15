# ---------------------------------------------------------------------------------------------------------------------
# fn_training_v2.jl is a MODIFICATION of training.jl from GeometricMachineLearning@v0.2.0
# Note: the original file is not part of the source code of GeometricMachineLearning.jl release package, instead it is 
# part of test setups in scripts/symplectic_autoencoders
#
# significant functionality changes to the original training.jl file are:
#   - system parameters are set in separate file fn_global_consts.jl
#   - training routine implements:
#       - epochwise training
#       - training with normalized data (Note: PSD encoder/decoder still works with unnormalized data to 
#         provide an unchanged comparison method error)
# ---------------------------------------------------------------------------------------------------------------------



println("1D WAVE EQUATION: TRAINING")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using GeometricMachineLearning 
using LinearAlgebra
using ProgressMeter
using Zygote
using HDF5
using CUDA
using GeometricIntegrators
using Plots
using KernelAbstractions

using OffsetArrays # needed in assemble_matrix_v2.jl
using ForwardDiff # needed in fn_initial_condition_v2.jl
using Random    #needed in fn_batch.jl
import GeometricMachineLearning: SystemType    #needed in fn_reduced_system_v2.jl

pythonplot()    # needed for .eps plots

include("../general/fn_assemble_matrix_v020.jl")
include("../general/fn_vector_fields_v020.jl")
include("../general/fn_initial_condition_v020.jl")
include("../../shared/fn_customtypes.jl")
include("../../shared/fn_loss.jl")
include("../../shared/fn_GML_custom/fn_batch_v020.jl")
include("../../shared/fn_GML_custom/fn_reduced_system_v020.jl")
include("../general/fn_global_consts.jl")

#endregion


#region Section 2: Helper functions

function _cpu_convert(ps::Tuple)
    output = ()
    for elem in ps 
        output = (output..., _cpu_convert(elem))
    end
    output
end

function _cpu_convert(ps::NamedTuple)
    output = ()
    for elem in ps
        output = (output..., _cpu_convert(elem))
    end
    NamedTuple{keys(ps)}(output)
end

_cpu_convert(A::AbstractArray) = Array(A)

_cpu_convert(Y::StiefelManifold) = StiefelManifold(_cpu_convert(Y.A))

function get_reduced_model(encoder, decoder;n=5, μ_val=0.51, Ñ=(N-2), n_time_steps=n_time_steps, integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    params = (μ=T_test(μ_val), Ñ=Ñ, Δx=T_test(1/(Ñ-1)))
    tstep = T_test(1/(n_time_steps-1))
    tspan = (T_test(0), T_test(1))
    ics = get_initial_condition_vector(params.μ, Ñ)
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field(v_field_explicit(params), decoder, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, tspan, tstep, ics; integrator=integrator, system_type=system_type)
end

# to work with normalized data
function get_reduced_model_normalized(encoder, decoder;n=5, μ_val=0.51, Ñ=(N-2), n_time_steps=n_time_steps, integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    params = (μ=T_test(μ_val), Ñ=Ñ, Δx=T_test(1/(Ñ-1)))
    tstep = T_test(1/(n_time_steps-1))
    tspan = (T_test(0), T_test(1))
    ics = get_initial_condition_vector(params.μ, Ñ)
    # added field x_ref
    zero = zeros(eltype(ics), size(ics))
    x_ref = ics - decoder(encoder(zero))
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field_normalized(v_field_explicit(params), decoder, x_ref, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, tspan, tstep, ics; x_ref=x_ref, integrator=integrator, system_type=system_type)
end

# draw batch to work with not-epochwise learning
function draw_batch!(batch::AbstractMatrix{T}, data::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}) where T
    batch_size = size(batch, 2)
    n_params = size(data, 2)
    param_indices = Int.(ceil.(rand(T, batch_size)*n_params))
    batch .= data[:, param_indices]
end

# plot the results
function plot_projection_reduction_errors(μ_errors, name_errors)
    number_errors = length(μ_errors[1])
    n_vals = zeros(Int, number_errors)
    nn_projection_vals = zeros(number_errors)
    nn_reduction_vals = zeros(number_errors)
    psd_projection_vals = zeros(number_errors)
    psd_reduction_vals = zeros(number_errors)
    for μ_key in keys(μ_errors)
        μ = string(μ_key)
        it = 0
        for n_key in keys(μ_errors[μ_key])
            it += 1
            n = parse(Int, string(n_key)[2:end])
            nn_projection_val = μ_errors[μ_key][n_key].projection_error.nn 
            nn_reduction_val = μ_errors[μ_key][n_key].reduction_error.nn 
            psd_projection_val = μ_errors[μ_key][n_key].projection_error.psd
            psd_reduction_val = μ_errors[μ_key][n_key].reduction_error.psd
            n_vals[it] = n 
            nn_projection_vals[it] = nn_projection_val 
            nn_reduction_vals[it] = nn_reduction_val 
            psd_projection_vals[it] = psd_projection_val
            psd_reduction_vals[it] = psd_reduction_val
        end
        plot_object = plot(n_vals, psd_projection_vals, color=:blue, seriestype=:scatter, markershape=:rect, ylimits=(0,y_limits_global), label="PSD projection", xlabel="reduced dimension n", ylabel="error", title=headline_error_plots_global, titlefontsize=11)
        plot!(plot_object, n_vals, psd_reduction_vals, color=:blue, seriestype=:scatter, label="PSD reduction")
        plot!(plot_object, n_vals, nn_projection_vals, color=:orange, seriestype=:scatter, markershape=:rect, label="NN projection")        
        plot!(plot_object, n_vals, nn_reduction_vals, color=:orange, seriestype=:scatter, label="NN reduction")
        savefig(plot_object, "plots_temp/$(name_errors)_$(μ[3:end]).eps")
    end
end

#endregion


#region Section 3: Initialize parameters...

println("Initializing parameters...")

T = T_learning_global
T_test = T_testing_global

opt = AdamOptimizer(T.((0.001, 0.9, 0.99, 1e-8))...)
retraction = Cayley()

if LossGML_global
    losstype = LossGML()
elseif LossModified_global
    losstype = LossModified()
else
    throw(GlobalConstantError("A loss function has to be chosen."))
end

n_epochs = n_epochs_global
batch_size = batch_size_global

n_time_steps = time_steps_global 
N = N_global + 2 

if plot_reconstructed_sol_global
    Ω = -0.5:(1/(N-1)):0.5
    time_int = 0:(1/(time_steps_global-1)):1
end

# setting some naming variables 
if TEST_global
    name_PSDerrors = "PSD_errors_N=$(N_global)_TEST"
    name_errors = "errors_N=$(N_global)_TEST"
else
    local n_range_string = "n="
    for n in n_range_global
        n_range_string = n_range_string * "_$n"
    end
    name_PSDerrors = "PSD_errors_N=$(N_global)_$(n_range_string)"
    name_errors = "errors_N=$(N_global)_$(n_range_string)"
end

plot_name = "reconstructed_NN"

if epochwise_global
    name_errors *= "_epochwise"
    plot_name *= "_epochwise"
end

if normalized_global 
    name_snapshot = "snapshot_matrix_normalized_N=$(N_global)"
    name_errors *= "_normalized"
    plot_name *= "_normalized"
else
    name_snapshot = "snapshot_matrix_N=$(N_global)"
end

if LossGML_global
    name_errors *= "_lossGML"
    plot_name *= "_lossGML"
else
    name_errors *= "_lossTheory"
    plot_name *= "_lossTheory"
end

#endregion


#region Section 4: Load all data required

println("Loading data...")

psd_data = h5open("resources/reproduction/$(name_PSDerrors).h5", "r") do file
        read(file, "errors")
    end
μ_range = h5open("resources/reproduction/$(name_PSDerrors).h5", "r") do file
        read(file, "mu_range")
    end
n_range = h5open("resources/reproduction/$(name_PSDerrors).h5", "r") do file
        read(file, "n_range")
    end

function gpu_backend()
    data = h5open("resources/reproduction/$(name_snapshot).h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("resources/reproduction/$(name_snapshot).h5", "r") do file
        read(file, "n_params")
    end
    (CUDABackend(), data |> cu, n_params)
end 

function cpu_backend()
    data = h5open("resources/reproduction/$(name_snapshot).h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("resources/reproduction/$(name_snapshot).h5", "r") do file
        read(file, "n_params")
    end
    (CPU(), data, n_params)
end 

backend, data, n_params = 
try 
    gpu_backend()
catch
    cpu_backend()
end

# Assert if the globals match with the data loaded
@assert n_time_steps == Int(size(data,2) / n_params)
@assert N == size(data,1)÷2

dl = DataLoader(data)
if epochwise_global
    batch = Batch(batch_size)
else
    batch = KernelAbstractions.allocate(backend,T, 2*N, batch_size)
end

#endregion


#region Section 5: Set up the network

println("Setting up the network...")

function get_nn_encoder_decoder(; n=5, n_epochs=500, activation=tanh, opt=opt, T=T)
    Ψᵉ = Chain(
        GradientLayerQ(2*N, 10*N, activation), 
        GradientLayerP(2*N, 10*N, activation),
        GradientLayerQ(2*N, 10*N, activation), 
        GradientLayerP(2*N, 10*N, activation),
        PSDLayer(2*N, 2*n; retraction=retraction)
    )
    Ψᵈ = Chain(
        GradientLayerQ(2*n, 10*n, activation), 
        GradientLayerP(2*n, 10*n, activation),
        PSDLayer(2*n, 2*N; retraction=retraction),
        GradientLayerQ(2*N, 2*N, activation)
         )
    model = Chain(  
                    Ψᵉ.layers..., 
                    Ψᵈ.layers...
    )
    ps = initialparameters(backend, T, model)
    optimizer_instance = Optimizer(opt, ps)
    epoch_losses = zeros(n_epochs)
    if epochwise_global
        progress_object = Progress(round(Int,dl.n_params*n_epochs/batch_size); enabled=true)
        for i in 1:n_epochs
            average_loss = optimize_for_one_epoch!(optimizer_instance, model, ps, dl, batch, losstype, progress_object)
            epoch_losses[i] = average_loss
        end
    else
        n_training_iterations = Int(ceil(n_epochs*dl.n_params/batch_size))
        progress_object = Progress(n_training_iterations; enabled=true)

        for _ in 1:n_training_iterations
            draw_batch!(batch, data)
            # this line shouldnt be needed, this is a zygote problem
            input_batch = copy(batch)
            loss_val, pb =  
                Zygote.pullback(ps -> loss(model, ps, input_batch, losstype), ps)
            dp = pb(one(loss_val))[1]
            try 
                optimization_step!(optimizer_instance, model, ps, dp)
            catch
                continue 
            end
            ProgressMeter.next!(progress_object; showvalues=[(:TrainingLoss, loss_val)])
        end
    end
    psᵉ = _cpu_convert(ps[1:length(Ψᵉ.layers)])
    psᵈ = _cpu_convert(ps[(length(Ψᵉ.layers)+1):end])   
    nn_encoder(z) = Ψᵉ(z, psᵉ)
    nn_decoder(ξ) = Ψᵈ(ξ, psᵈ)

    nn_encoder, nn_decoder, epoch_losses
end

function get_enocders_decoders(n_range)
    encoders_decoders = NamedTuple()
    epoch_losses= zeros(T, length(n_range), n_epochs) 
    for (n_ind, n) in zip(1:length(n_range), n_range)
        println("   Learning encoder/decoder for reduced dimension n=$n...")
        nn_encoder, nn_decoder, losses = get_nn_encoder_decoder(n=n, n_epochs=n_epochs)
        nn_ed = (encoder=nn_encoder, decoder=nn_decoder)
        encoders_decoders = NamedTuple{(keys(encoders_decoders)..., Symbol("n"*string(n)))}((values(encoders_decoders)..., nn_ed))
        if epochwise_global
            epoch_losses[n_ind, :] = losses
        end
    end
    encoders_decoders, epoch_losses
end

#endregion


#region Section 6: Code execution


print("Perform the learning: ")
if epochwise_global
    print("epochwise ")
else
    print("not epochwise")
end
if normalized_global
    print("+ normalized ")
else 
    print("+ unnormalized ")
end
if LossGML_global
    println("+ lossGML...")
else
    println("+ lossTheory...")
end
encoder_decoder_time = 
@timed begin
    encoders_decoders, epoch_losses = get_enocders_decoders(n_range)
end 

println("Calculating the errors...")
error_time =
@timed begin
    μ_errors = NamedTuple()
    for (μ_ind, μ_test_val) in zip(1:length(μ_range), μ_range)
        println("   I am at μ=$μ_test_val")
        dummy_rs = get_reduced_model(nothing, nothing; n=1, μ_val=μ_test_val, Ñ=(N-2))
        sol_full = perform_integration_full(dummy_rs)
        errors = NamedTuple()
        for (n_ind, n) in zip(1:length(n_range),n_range)
            println("       I am at reduced dimension n=$n")
            current_n_identifier = Symbol("n"*string(n))
            nn_encoder, nn_decoder = encoders_decoders[current_n_identifier]
            if normalized_global
                nn_rs = get_reduced_model_normalized(nn_encoder, nn_decoder; n=n, μ_val=μ_test_val, Ñ=(N-2))
                red_err_nn, int_time, sol_matrix_reconst = compute_reduction_error_normalized(nn_rs, sol_full; reconst_sol=plot_reconstructed_sol_global)
                proj_err_nn = compute_projection_error_normalized(nn_rs, sol_full)
            else
                nn_rs = get_reduced_model(nn_encoder, nn_decoder; n=n, μ_val=μ_test_val, Ñ=(N-2))
                red_err_nn, int_time, sol_matrix_reconst = compute_reduction_error(nn_rs, sol_full; reconst_sol=plot_reconstructed_sol_global)
                proj_err_nn = compute_projection_error(nn_rs, sol_full)
            end
            println("       Reduction error: $(red_err_nn)")
            println("       Integration time: $(int_time.time)")
            if plot_reconstructed_sol_global
                plot_object = Plots.surface(Ω,time_int,transpose(sol_matrix_reconst[1:length(Ω),:]), xlabel="ξ", ylabel="t", title="Reconstructed solution for Ω=(-0.5,0.5), I=(0,1), N=$(N)", titlefontsize=12)
                savefig(plot_object, "plots_temp/$(plot_name)_N=$(N_global)_n=$(n)_μ=$(μ_test_val).eps")
            end
            reduction_errors = (psd=psd_data[μ_ind, n_ind, 1], nn=red_err_nn)
            projection_errors = (psd=psd_data[μ_ind, n_ind, 2], nn=proj_err_nn)
            temp_errors = (reduction_error=reduction_errors, projection_error=projection_errors)
            errors = NamedTuple{(keys(errors)..., current_n_identifier)}((values(errors)..., temp_errors))
        end
        global μ_errors = NamedTuple{(keys(μ_errors)..., Symbol("μ"*string(μ_test_val)))}((values(μ_errors)..., errors))
    end   
end

println("Plotting the errors...")
plot_time =
@timed begin
    plot_projection_reduction_errors(μ_errors, name_errors)
end

println("Writing the log data...")
open("plots_temp/timeinfo.txt", "w") do file

    write(file, "Parameter setup: \n\n")

    if normalized_global
        write(file, "Normalized: Yes \n\n")
    else 
        write(file, "Normalized: No \n\n")
    end
    if epochwise_global
        write(file, "Epochwise: Yes \n")
        write(file, "Number epochs: $n_epochs_global \n")
        write(file, "Batch size: $batch_size_global \n\n")
    else
        write(file, "Epochwise: No \n\n")
    end
    write(file, "Full dimension N: $N_global \n")
    write(file, "Reduced dimensions n: $n_range \n")
    write(file, "Time steps: $time_steps_global \n")
    write(file, "Domain: Ω=(-0.5,0.5), I=(0,1) \n")
    write(file, "number training set μ-s: $μ_range \n")
    write(file, "μ_testing_range: $μ_range \n\n")


    write(file, "DataType training data (and encoder/decoder): $T_learning_global \n")
    write(file, "DataType for testing: $T_testing_global \n\n\n")

    
    write(file, "Log data: \n\n")

    encoder_decoder_time_str = string("Time (seconds): ", encoder_decoder_time.time, "\n")
    encoder_decoder_allocations_str = string("Allocations (bytes): ", encoder_decoder_time.bytes, "\n")
    encoder_decoder_gctime_str = string("GC time (seconds): ", encoder_decoder_time.gctime, "\n\n")

    write(file, "Encoder_Decoder calculation: \n")
    write(file, encoder_decoder_time_str)
    write(file, encoder_decoder_allocations_str)
    write(file, encoder_decoder_gctime_str)
    
    error_time_str = string("Time (seconds): ", error_time.time, "\n")
    error_allocations_str = string("Allocations (bytes): ", error_time.bytes, "\n")
    error_gctime_str = string("GC time (seconds): ", error_time.gctime, "\n\n")

    write(file, "Error calculation: \n")
    write(file, error_time_str)
    write(file, error_allocations_str)
    write(file, error_gctime_str)

    plot_time_str = string("Time (seconds): ", plot_time.time, "\n")
    plot_allocations_str = string("Allocations (bytes): ", plot_time.bytes, "\n")
    plot_gctime_str = string("GC time (seconds): ", plot_time.gctime, "\n")

    write(file, "Plot creation: \n")
    write(file, plot_time_str)
    write(file, plot_allocations_str)
    write(file, plot_gctime_str)
end


μ_len =length(μ_range)
n_len = length(n_range) 
error_mat = zeros(μ_len, n_len, 2)
for (μ_ind,μ_key) in zip(1:μ_len,keys(μ_errors))
    for (n_ind, n_key) in zip(1:n_len,keys(μ_errors[μ_key]))
        error_mat[μ_ind,n_ind,1] = μ_errors[μ_key][n_key].reduction_error.nn 
        error_mat[μ_ind,n_ind,2] = μ_errors[μ_key][n_key].projection_error.nn 
    end
end

println("Saving the results...")
h5open("resources/reproduction/$(name_errors).h5", "w") do h5
    h5["nn_errors"] = error_mat
    h5["psd_errors"] = psd_data
    h5["nu_test_range"] = μ_range
    h5["n_range"] = n_range
    if epochwise_global & save_epoch_loss_global
        h5["epoch_losses"] = epoch_losses
    end
end 

#endregion