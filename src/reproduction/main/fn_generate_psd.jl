# ---------------------------------------------------------------------------------------------------------------------
# fn_generate_psd.jl is a SELF-CREATED file to precalculate the psd errors
# ---------------------------------------------------------------------------------------------------------------------



println("1D WAVE EQUATION: GENERATE PSD ERRORS")
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
using HDF5

using OffsetArrays # needed in assemble_matrix_v2.jl
using ForwardDiff # needed in fn_initial_condition_v2.jl
using Random    #needed in fn_batch.jl
import GeometricMachineLearning: SystemType    #needed in fn_reduced_system_v2.jl

pythonplot()    # needed for .eps plots

include("../general/fn_assemble_matrix_v020.jl")
include("../general/fn_vector_fields_v020.jl")
include("../general/fn_initial_condition_v020.jl")
include("../../shared/fn_customtypes.jl")
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

function get_reduced_model(encoder, decoder;n=5, μ_val=0.51, Ñ=(N-2), n_time_steps=n_time_steps, integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    params = (μ=μ_val, Ñ=Ñ, Δx=T(1/(Ñ-1)))
    tstep = T(1/(n_time_steps-1))
    tspan = (T(0), T(1))
    ics = get_initial_condition_vector(μ_val, Ñ)
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field_LINEAR(v_field_explicit(params), decoder, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, tspan, tstep, ics; integrator=integrator, system_type=system_type)
end

function get_psd_encoder_decoder(; n=5)
    Φ = svd(hcat(data[1:N,:], data[(N+1):2*N,:])).U[:,1:n]
    PSD = hcat(vcat(Φ, zero(Φ)), vcat(zero(Φ), Φ))

    PSD_cpu = _cpu_convert(PSD)
    psd_encoder(z) = PSD_cpu'*z 
    psd_decoder(ξ) = PSD_cpu*ξ
    psd_encoder, psd_decoder
end

function get_enocders_decoders(n_range)
    encoders_decoders = NamedTuple()
    for n in n_range
        psd_encoder, psd_decoder = get_psd_encoder_decoder(n=n)
        psd_ed = (encoder=psd_encoder, decoder=psd_decoder)
        encoders_decoders_current = psd_ed
        encoders_decoders = NamedTuple{(keys(encoders_decoders)..., Symbol("n"*string(n)))}((values(encoders_decoders)..., encoders_decoders_current))
    end
    encoders_decoders
end

#endregion


#region Section 3: Initialize parameters 

println("Initializing parameters...")

T = T_testing_global
n_epochs = n_epochs_global
n_range = n_range_global
N = N_global + 2
sys_dim = 2*N
n_time_steps = time_steps_global

if plot_reconstructed_sol_global
    Ω = -0.5:(1/(N-1)):0.5
    time_int = 0:(1/(time_steps_global-1)):1
end

if p_zero_global
    name = "snapshot_matrix_pzero_N=$(N_global)"
else
    name = "snapshot_matrix_N=$(N_global)"
end

if TEST_global
    μ_range = (T(μ_testing_TEST_global))  
    name_muerrs = "PSD_errors_N=$(N_global)_TEST"
else
    μ_range = T.(μ_testing_range_global)  
    local n_range_string = "_n="
    for n in n_range_global
        n_range_string = n_range_string * "_$n"
    end
    name_muerrs = "PSD_errors_N=$(N_global)" * "$n_range_string"
end

#endregion


#region Section 4: Load all data required

println("Loading data...")

function gpu_backend()
    data = h5open("resources/reproduction/$(name).h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("resources/reproduction/$(name).h5", "r") do file
        read(file, "n_params")
    end
    (CUDABackend(), data |> cu, n_params)
end 

function cpu_backend()
    data = h5open("resources/reproduction/$(name).h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("resources/reproduction/$(name).h5", "r") do file
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

data_cpu = _cpu_convert(data)

# Assert if the globals match with the data loaded
@assert sys_dim == size(data,1)
@assert n_time_steps == Int(size(data,2) / n_params)
@assert N == size(data,1)÷2

#endregion


#region Section 5: Code execution

encoders_decoders = get_enocders_decoders(n_range)

println("Calculating the errors...")
μ_errors = NamedTuple()
for μ_test_val in μ_range
    println("   I am at μ=$μ_test_val")
    dummy_rs = get_reduced_model(nothing, nothing; n=1, μ_val=μ_test_val, Ñ=(N-2))
    sol_full = perform_integration_full(dummy_rs)
    errors = NamedTuple()
    for n in n_range
        println("       I am at n=$n")
        current_n_identifier = Symbol("n"*string(n))

        psd_encoder, psd_decoder = encoders_decoders[current_n_identifier]

        psd_rs = get_reduced_model(psd_encoder, psd_decoder; n=n, μ_val=μ_test_val, Ñ=(N-2))

        psd_red, int_time, sol_matrix_reconst = compute_reduction_error(psd_rs, sol_full; reconst_sol=plot_reconstructed_sol_global)
        psd_proj=compute_projection_error(psd_rs, sol_full)
        println("       Reduction error: $(psd_red)")
        println("       Integration time: $(int_time.time)")
        if plot_reconstructed_sol_global
            plot_object = Plots.surface(Ω,time_int,transpose(sol_matrix_reconst[1:length(Ω),:]), xlabel="ξ", ylabel="t", title="Reconstructed solution for Ω=(-0.5,0.5), I=(0,1), N=$(N)", titlefontsize=12)
            if normalized_global
                savefig(plot_object, "plots_temp/reconstructed_PSD_normalized_N=$(N_global)_n=$(n)_μ=$(μ_test_val).eps")
            else
                savefig(plot_object, "plots_temp/reconstructed_PSD_N=$(N_global)_n=$(n)_μ=$(μ_test_val).eps")
            end
        end
        temp_errors = (psd_red, psd_proj)
        errors = NamedTuple{(keys(errors)..., current_n_identifier)}((values(errors)..., temp_errors))
    end
    global μ_errors = NamedTuple{(keys(μ_errors)..., Symbol("μ"*string(μ_test_val)))}((values(μ_errors)..., errors))
end

μ_len =length(μ_range)
n_len = length(n_range) 
error_mat = zeros(μ_len, n_len, 2)
for (μ_ind,μ_key) in zip(1:μ_len,keys(μ_errors))
    for (n_ind, n_key) in zip(1:n_len,keys(μ_errors[μ_key]))
        error_mat[μ_ind,n_ind,:] .= μ_errors[μ_key][n_key]
    end
end

h5open("resources/reproduction/$(name_muerrs).h5", "w") do h5
    h5["errors"] = error_mat
    h5["mu_range"] = collect(μ_range)
    h5["n_range"] = collect(n_range)
end

#endregion
