# ---------------------------------------------------------------------------------------------------------------------
# fn_training_v020_alternative_optimizer_sineGordon_LINEARNETWORK.jl is a MODIFICATION of training.jl from GeometricMachineLearning@v0.2.0
# Note: the original file is not part of the source code of GeometricMachineLearning.jl release package, instead it is 
# part of test setups in scripts/symplectic_autoencoders
#
# significant functionality changes to the original training.jl file are:
#   - implements training of sine Gordon equation (sG) instead of wave equation
#   - system parameters are set in separate file fn_global_consts.jl (included in subproject sineGordon)
#   - training routine implements:
#       - epochwise training
#       - training with normalized data (Note: PSD encoder/decoder still works with unnormalized data to 
#         provide an unchanged comparison method error)
#       - use of SELF-IMPLEMENTED optimizers 
#       - changed network layers: only linear PSDLayers
# ---------------------------------------------------------------------------------------------------------------------



println("1D SINE GORDON EQUATION: ALTERNATIVE OPTIMIZER TRAINING OF LINEAR NETWORK")
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
import GeometricMachineLearning: LayerWithManifold  #needed in fn_retractions_v2.jl
import GeometricMachineLearning: SystemType    #needed in fn_reduced_system_v2.jl

using AbstractNeuralNetworks: AbstractExplicitLayer, AbstractExplicitCell # needed in fn_optimizer_v2.jl

pythonplot()    # needed for .eps plots

include("../../shared/fn_customtypes.jl")
include("../../shared/fn_loss.jl")

include("../fn_adam_modified_scripts/fn_utils_v020.jl")
include("../fn_adam_modified_scripts/fn_retractions_v020.jl")
include("../fn_adam_modified_scripts/fn_stiefel_manifold_v020.jl")

include("../fn_adam_added_scripts/fn_adam_optimizer_stiefel.jl")
include("../fn_adam_added_scripts/fn_adam_optimizer_stiefel_withDecay.jl")
include("../fn_adam_added_scripts/fn_stochastic_gradient_descent_with_momentum.jl")
include("../fn_adam_added_scripts/fn_stochastic_gradient_descent_with_momentum_withDecay.jl")
include("../fn_adam_added_scripts/fn_cayley_adam_from_other_paper.jl")

include("../fn_adam_modified_scripts/fn_optimizer_v020.jl")
include("../fn_adam_modified_scripts/fn_init_optimizer_cache_v020.jl")

include("../general/fn_global_consts_alternative_optimizer_specific.jl")

include("../../shared/fn_GML_custom/fn_reduced_system_v020.jl")
include("../../shared/fn_GML_custom/fn_batch_v020.jl")

include("../../sineGordon/general/fn_hamiltonian_system.jl")
include("../../sineGordon/general/fn_math_utils.jl")
include("../../sineGordon/general/fn_global_consts.jl")

include("../../sineGordon/examples/fn_sine_gordon.jl")
include("../../sineGordon/examples/fn_example1.jl")
include("../../sineGordon/examples/fn_single_soliton.jl")
include("../../sineGordon/examples/fn_soliton_antisoliton_breather.jl")
include("../../sineGordon/examples/fn_soliton_soliton_doublets.jl")

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

# create an instance of the reduced model as specified in fn_reduced_system_v2.jl
function get_reduced_model(T::DataType, encoder, decoder, n, ν_val, N, num_time_points, Ω, time_int, sG::SineGordon; integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    a = T(first(Ω))
    b = T(last(Ω))
    t0 = T(first(time_int))
    t1 = T(last(time_int))
    ν =T(ν_val) 
    params = (ν=ν, N=N, h_N=T((b-a)/(N+1)), φ=(t -> sG.φ(t,ν_val,a)), ψ=(t -> sG.ψ(t,ν_val,b)))
    tstep = T((t1-t0)/(num_time_points-1))
    tspan = (t0, t1)
    ics = vcat(T.(sG.u₀.(T.(Ω),ν)), T.(sG.u₁.(T.(Ω),ν)))
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field_LINEAR(v_field_explicit(params), decoder, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, tspan, tstep, ics; integrator=integrator, system_type=system_type)
end

# create an instance of the reduced model as specified in fn_reduced_system_v2.jl
function get_reduced_model_normalized(T::DataType, encoder, decoder, n, ν_val, N, num_time_points, Ω, time_int, sG::SineGordon; integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    a = T(first(Ω))
    b = T(last(Ω))
    t0 = T(first(time_int))
    t1 = T(last(time_int))
    ν =T(ν_val) 
    params = (ν=ν, N=N, h_N=T((b-a)/(N+1)), φ=(t -> sG.φ(t,ν_val,a)), ψ=(t -> sG.ψ(t,ν_val,b)))
    tstep = T((t1-t0)/(num_time_points-1))
    tspan = (t0, t1)
    ics = vcat(T.(sG.u₀.(T.(Ω),ν)), T.(sG.u₁.(T.(Ω),ν)))
    # CHANGE fn: added field x_ref
    zero = zeros(eltype(ics), size(ics))
    x_ref = ics - decoder(encoder(zero))
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field_normalized_LINEAR(v_field_explicit(params), decoder, x_ref, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, tspan, tstep, ics; x_ref=x_ref, integrator=integrator, system_type=system_type)
end

# generate exact solution of specified data type
function generate_exact_solution(T::DataType, ν_val, Ω, time_int, sG::SineGordon)
    t = length(time_int)
    l = length(Ω)
    sol_matrix = zeros(T, 2*l, t)
    for it in zip(axes(time_int, 1), time_int)
        i = it[1]; t = it[2]
        for it2 in zip(axes(Ω,1),Ω)
            k=it2[1]; ξ=it2[2]
            sol_matrix[k, i] = T(sG.exact_solution(t,ξ,ν_val))
            sol_matrix[k + l, i] = T(ForwardDiff.derivative(t -> sG.exact_solution(t,ξ,ν_val), t))
        end
    end
    return sol_matrix
end

# plot the results
function plot_projection_reduction_errors(ν_errors, a, b, t0, t1, N, name_errors)
    number_errors = length(ν_errors[1])
    n_vals = zeros(Int, number_errors)
    nn_projection_vals = zeros(number_errors)
    nn_reduction_vals = zeros(number_errors)
    psd_projection_vals = zeros(number_errors)
    psd_reduction_vals = zeros(number_errors)
    for ν_key in keys(ν_errors)
        ν = string(ν_key)
        it = 0
        for n_key in keys(ν_errors[ν_key])
            it += 1
            n = parse(Int, string(n_key)[2:end])
            nn_projection_val = ν_errors[ν_key][n_key].projection_error.nn 
            nn_reduction_val = ν_errors[ν_key][n_key].reduction_error.nn 
            psd_projection_val = ν_errors[ν_key][n_key].projection_error.psd
            psd_reduction_val = ν_errors[ν_key][n_key].reduction_error.psd
            n_vals[it] = n 
            nn_projection_vals[it] = nn_projection_val 
            nn_reduction_vals[it] = nn_reduction_val 
            psd_projection_vals[it] = psd_projection_val
            psd_reduction_vals[it] = psd_reduction_val
        end
        plot_object = plot(n_vals, psd_projection_vals, color=:blue, seriestype=:scatter, markershape=:rect, label="PSD projection", xlabel="reduced dimension n", ylabel="error", title="Normalized errors for Ω=($a,$b), I=($t0,$t1), N=$N", titlefontsize=11)
        plot!(plot_object, n_vals, psd_reduction_vals, color=:blue, seriestype=:scatter, label="PSD reduction")
        plot!(plot_object, n_vals, nn_projection_vals, color=:orange, seriestype=:scatter, markershape=:rect, label="NN projection")        
        plot!(plot_object, n_vals, nn_reduction_vals, color=:orange, seriestype=:scatter, label="NN reduction")
        savefig(plot_object, "plots_temp/$(name_errors)_ν=$(ν[3:end]).eps")
    end
end

#endregion


#region Section 3: Initialize parameters 

println("Initializing parameters...")

# specify sG example
example = example_in_use_global

if example == "example1"
    sG_constructor = SineGordon_Example1
elseif example == "single_soliton"
    sG_constructor = SineGordon_SingleSoliton
elseif example == "soliton_antisoliton_breather"
    sG_constructor = SineGordon_SolitonAntisolitonBreather
elseif example == "soliton_soliton_doublets"
    sG_constructor = SineGordon_SolitonSolitonDoublet
else
    throw(ExampleNotFoundError("Example '$example' not found"))
end    

# instantiate sG object, note: only necessary values are assigned, others (included in data, omega, time, nu_range)
# are omitted (set to nothing)
sG = sG_constructor(
   nothing,
   nothing,
   nothing,
   nothing,
   nothing;
   T_analytic=T_analytic_global,
   N=nothing,
   t_steps=nothing)
 
if StiefelAdamOptimizer_global
    opt = StiefelAdamOptimizer(sG.T_analytic.((0.001, 0.9, 0.99, 1e-8))...)
    opt_name = "StiefelAdamOptimizer"
elseif StiefelAdamOptimizerWithDecay_global
    opt = StiefelAdamOptimizerWithDecay(sG.T_analytic.((0.001, 0.9, 0.9, 0.99, 0.99, 1e-8))...) 
    opt_name = "StiefelAdamOptimizerWithDecay"
elseif StiefelStochasticGradientDescentWithMomentum_global
    opt = StiefelStochasticGradientDescentWithMomentum(sG.T_analytic.((0.001, 0.9))...)
    opt_name = "StiefelStochasticGradientDescentWithMomentum"
elseif StiefelStochasticGradientDescentWithMomentumWithDecay_global
    opt = StiefelStochasticGradientDescentWithMomentumWithDecay(sG.T_analytic.((0.001, 0.9))...)
    opt_name = "StiefelStochasticGradientDescentWithMomentumWithDecay"
elseif StiefelCayleyAdamOptimizerFromOtherPaper_global
    opt = StiefelCayleyAdamOptimizerFromOtherPaper(sG.T_analytic.((0.4, 0.9, 0.9, 0.99, 0.99, 1e-8, 1.0, 0.5, 2))...)
    opt_name = "StiefelCayleyAdamOptimizerFromOtherPaper"
else
    # if non is selected, throw optimizer not selected exception
    throw(NoOptimizerSelected("An optimizer has to  be selected."))
end

if LossGML_global
    losstype = LossGML()
elseif LossModified_global
    losstype = LossModified()
else
    throw(GlobalConstantError("A loss function has to be chosen."))
end
opt_var=""
if CanonicalMetric_global
    met = Canonical()
    opt_var = "can"
elseif EuclideanMetric_global
    met = Euclidean()
    opt_var= "eucl"
else
    throw(GlobalConstantError("A metric has to be chosen."))
end

if SubmanifoldVectorTransport_global
    vt = SubmanifoldVectortransport()
    opt_var *= ",sub"
elseif DifferentialVectorTransport_global
    vt = DifferentialVectortransport()
    opt_var *= ",diff"
else
    throw(GlobalConstantError("A vector transport has to be chosen."))
end

retraction = Cayley()

batch_size = batch_size_global
n_epochs = n_epochs_global

num_time_points = time_steps_global + 1
N = N_global
sys_dim = 2*N

# setting some naming variables 
if TEST_global
    name_PSDerrors = "PSD_errors_N=$(N_global)_TEST"
    name_errors = "alternative_optimizer_$(example)_errors_N=$(N_global)_TEST"
else
    local n_range_string = "n="
    for n in range_n_global
        n_range_string = n_range_string * "_$n"
    end
    if PSDnormalized_global # should be set to false as we do not want to change the comparison method
        name_PSDerrors = "PSD_errors_normalized_N=$(N_global)_$(n_range_string)"
    else
        name_PSDerrors = "PSD_errors_N=$(N_global)_$(n_range_string)"
    end
    name_errors = "alternative_optimizer_$(example)_errors_N=$(N_global)_$(n_range_string)_epochwise"
end

plot_name = "alternative_optimizer_reconstructed_NN_epochwise"

if normalized_global 
    name_snapshot = "snapshots_normalized_N=$(N_global)"
    name_errors *= "_normalized"
    plot_name *= "_normalized"
else
    name_snapshot = "snapshots_N=$(N_global)"
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

# load training data for specified example 
raw_data=h5open("resources/sineGordon/data/$(example)_$(name_snapshot).h5", "r") do file
    read(file,"data")
end
raw_omega=h5open("resources/sineGordon/data/$(example)_$(name_snapshot).h5", "r") do file
    read(file,"omega")
end
time=h5open("resources/sineGordon/data/$(example)_$(name_snapshot).h5", "r") do file
    read(file,"time")
end
nu_range=h5open("resources/sineGordon/data/$(example)_$(name_snapshot).h5", "r") do file
    read(file,"nu_range")
end

# load precalculated PSD-errors
psd_data = h5open("resources/sineGordon/data/$(example)_$(name_PSDerrors).h5", "r") do file
        read(file, "psd_errors")
    end
ν_testing_range = h5open("resources/sineGordon/data/$(example)_$(name_PSDerrors).h5", "r") do file
        read(file, "nu_test_range")
    end
n_range = h5open("resources/sineGordon/data/$(example)_$(name_PSDerrors).h5", "r") do file
        read(file, "n_range")
    end

#reshape omega: cut out edges as they are not part of the disrectized hamiltonian, NOTE: this is different from the
# the initial wave equation example where the hamiltonian dimension was (2N + 2), here we have 2N!!!
omega = raw_omega[2:lastindex(raw_omega) - 1]
num_νs = length(nu_range)

# Assert if the globals match with the data loaded
@assert N == length(omega)
@assert n_range == range_n_global
@assert num_time_points == length(time)
@assert sG.T_analytic == eltype(raw_data)

# reshape data; IMPORTANT: cut out q_0, q_{N+1},p_0, p_{N+1} as learned system is 
# of size 2N NOT 2(N+2) other than before with the wave equation!
data_reshapedandsmall = zeros(2*N,num_time_points*num_νs)
for i in 0:num_νs-1
    start_ind = i*num_time_points + 1
    end_ind = (i+1)*num_time_points
    data_reshapedandsmall[1:N,start_ind:end_ind] = raw_data[2:N+1,:,i+1]
    data_reshapedandsmall[N+1:2*N,start_ind:end_ind] = raw_data[N+4:2*(N+2)-1,:,i+1]
end

# instantiate backend and load data to GPU if any
backend, data = 
try 
    (CUDABackend(), data_reshapedandsmall |> cu)
catch
    (CPU(), data_reshapedandsmall)
end

# instantiate DataLoader object
dl = DataLoader(data)

#endregion


#region Section 5: Set up the network

println("Setting up the network...")

# logic for the whole network model and the endcoder/decoder learning
function get_nn_encoder_decoder(; n=5, n_epochs=500, activation=tanh, opt=opt, T=sG.T_analytic)
    # ensuring that step length is reset for each new encoder/decoder learning (important for decay optimizers)
    if StiefelCayleyAdamOptimizerFromOtherPaper_global
        opt.η = 0.4
    else
        opt.η = 0.001
    end
    Ψᵉ = Chain(
        PSDLayer(2*N, 2*n; retraction=retraction))
    Ψᵈ = Chain(
        PSDLayer(2*n, 2*N; retraction=retraction))
    model = Chain(  
                    Ψᵉ.layers..., 
                    Ψᵈ.layers...)
    ps = initialparameters(backend, T, model)
    optimizer_instance = Optimizer(opt, ps)

    # perform the learning
    epoch_losses = zeros(n_epochs)
    progress_object = Progress(round(Int,dl.n_params*n_epochs/batch_size); enabled=true)
    batch = Batch(batch_size)
    for i in 1:n_epochs
        average_loss = optimize_for_one_epoch!(optimizer_instance, met, vt, model, ps, dl, batch, losstype, progress_object)
        epoch_losses[i] = average_loss
    end

    psᵉ = _cpu_convert(ps[1:length(Ψᵉ.layers)])
    psᵈ = _cpu_convert(ps[(length(Ψᵉ.layers)+1):end])   
    nn_encoder(z) = Ψᵉ(z, psᵉ)
    nn_decoder(ξ) = Ψᵈ(ξ, psᵈ)
    
    nn_encoder, nn_decoder, epoch_losses
end


# logic to get the encoder/decoder
function get_enocders_decoders(n_range)
    encoders_decoders = NamedTuple()
    epoch_losses = zeros(T, length(n_range), n_epochs) 
    for (n_ind, n) in zip(1:length(n_range), n_range)
        println("   Learning encoder/decoder for reduced dimension n=$n...")
        nn_encoder, nn_decoder, losses = get_nn_encoder_decoder(n=n, n_epochs=n_epochs)
        nn_ed = (encoder=nn_encoder, decoder=nn_decoder)
        encoders_decoders = NamedTuple{(keys(encoders_decoders)..., Symbol("n"*string(n)))}((values(encoders_decoders)..., nn_ed))
        epoch_losses[n_ind, :] = losses
    end
    encoders_decoders, epoch_losses
end

#endregion


#region Section 6: Code execution

# specify datatype to be used for testing, should globally be specified in global_consts.jl to have the same testing type 
# in all code parts (including the precalculated psd errors!)
T = T_testing_global 
omega_T = T.(omega)
time_T = T.(time)

print("Perform the learning: $(opt_name)($(opt_var)) ")
if normalized_global
    print("+ epochwise + normalized ")
else 
    print("+ epochwise + unnormalized ")
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
    ν_errors = NamedTuple()
    for (ν_ind, ν_test_val) in zip(1:length(ν_testing_range),ν_testing_range)
        println("   I am at ν=$ν_test_val")
        sol_matrix_full = generate_exact_solution(T, ν_test_val, omega_T, time_T, sG)
        errors = NamedTuple()
        for (n_ind, n) in zip(1:length(n_range),n_range)
            println("       I am at reduced dimension $n")
            current_n_identifier = Symbol("n"*string(n))
            nn_encoder, nn_decoder = encoders_decoders[current_n_identifier]
            if normalized_global
                nn_rs = get_reduced_model_normalized(T, nn_encoder, nn_decoder, n, ν_test_val, N, num_time_points, omega, time, sG)
                nn_red, time_needed, sol_matrix_reconst = compute_reduction_error_analytic_normalized(nn_rs, sol_matrix_full; reconst_sol=plot_reconstructed_sol_global)
                nn_proj=compute_projection_error_analytic_normalized(nn_rs, sol_matrix_full)
            else
                nn_rs = get_reduced_model(T, nn_encoder, nn_decoder, n, ν_test_val, N, num_time_points, omega, time, sG)
                nn_red, time_needed, sol_matrix_reconst = compute_reduction_error_analytic(nn_rs, sol_matrix_full; reconst_sol=plot_reconstructed_sol_global)
                nn_proj=compute_projection_error_analytic(nn_rs, sol_matrix_full)
            end
            println("       Reduction error: $(nn_red)")
            println("       Integration time: $(time_needed.time)")
            if plot_reconstructed_sol_global
                plot_object = Plots.surface(omega_T, time_T,transpose(sol_matrix_reconst[1:length(omega_T),:]), xlabel="ξ", ylabel="t", title="Reconstructed solution for Ω=($(a_global),$(b_global)), I=($(start_time_global),$(end_time_global)), N=$(N_global)", titlefontsize=12)
                savefig(plot_object, "plots_temp/$(plot_name)_N=$(N_global)_n=$(n)_ν=$(ν_test_val).eps")
            end
            reduction_errors = (psd=psd_data[ν_ind, n_ind, 1], nn=nn_red)
            projection_errors = (psd=psd_data[ν_ind, n_ind, 2], nn=nn_proj)
            temp_errors = (reduction_error=reduction_errors, projection_error=projection_errors)
            errors = NamedTuple{(keys(errors)..., current_n_identifier)}((values(errors)..., temp_errors))
        end
        global ν_errors = NamedTuple{(keys(ν_errors)..., Symbol("ν"*string(ν_test_val)))}((values(ν_errors)..., errors))
    end
end

println("Plotting the errors...")
plot_time_errors =
@timed begin
    plot_projection_reduction_errors(ν_errors, first(raw_omega), last(raw_omega), first(time), last(time), N, name_errors)
end

println("Writing the log data...")
open("plots_temp/timeinfo.txt", "w") do file

    write(file, "Parameter setup: \n\n")

    if normalized_global
        write(file, "Normalized: Yes \n\n")
    else 
        write(file, "Normalized: No \n\n")
    end

    write(file, "Epochwise: Yes \n")
    write(file, "Number epochs: $n_epochs_global \n")
    write(file, "Batch size: $batch_size_global \n\n")

    write(file, "Full dimension N: $N_global \n")
    write(file, "Reduced dimensions n: $range_n_global \n")
    write(file, "Time steps: $time_steps_global \n")
    write(file, "Domain: Ω=($a_global, $b_global), I=($start_time_global, $end_time_global) \n")
    write(file, "number training set ν-s: $nu_range \n")
    write(file, "ν_testing_range: $ν_testing_range_global \n")

    write(file, "DataType training data (and encoder/decoder): $(sG.T_analytic) \n")
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

    plot_errors_time_str = string("Time (seconds): ", plot_time_errors.time, "\n")
    plot_errors_allocations_str = string("Allocations (bytes): ", plot_time_errors.bytes, "\n")
    plot_errors_gctime_str = string("GC time (seconds): ", plot_time_errors.gctime, "\n")

    write(file, "Error plot creation: \n")
    write(file, plot_errors_time_str)
    write(file, plot_errors_allocations_str)
    write(file, plot_errors_gctime_str)

end


ν_len =length(ν_testing_range)
n_len = length(n_range) 
error_mat = zeros(ν_len, n_len, 2)
for (ν_ind,ν_key) in zip(1:ν_len,keys(ν_errors))
    for (n_ind, n_key) in zip(1:n_len,keys(ν_errors[ν_key]))
        error_mat[ν_ind,n_ind,1] = ν_errors[ν_key][n_key].reduction_error.nn 
        error_mat[ν_ind,n_ind,2] = ν_errors[ν_key][n_key].projection_error.nn 
    end
end

h5open("resources/sineGordon/data/$(name_errors).h5", "w") do h5
    h5["nn_errors"] = error_mat
    h5["psd_errors"] = psd_data
    h5["nu_test_range"] = ν_testing_range
    h5["n_range"] = n_range
    if save_epoch_loss_global
        h5["epoch_losses"] = epoch_losses
    end
end 

#endregion