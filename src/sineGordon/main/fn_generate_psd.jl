# ---------------------------------------------------------------------------------------------------------------------
# fn_generate_psd.jl is a SELF-CREATED file to precalculate the psd errors
# ---------------------------------------------------------------------------------------------------------------------



println("1D SINE GORDON EQUATION: PSD ERROR CALCULATION")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using GeometricMachineLearning 
using LinearAlgebra
using HDF5
using GeometricIntegrators
using Plots

using ForwardDiff # needed in sine_gordon.jl
import GeometricMachineLearning: SystemType    #needed in fn_reduced_system_v2.jl

pythonplot()

include("../general/fn_hamiltonian_system.jl")
include("../../shared/fn_GML_custom/fn_reduced_system_v020.jl")
include("../general/fn_math_utils.jl")
include("../general/fn_global_consts.jl")

include("../examples/fn_sine_gordon.jl")
include("../examples/fn_example1.jl")
include("../examples/fn_single_soliton.jl")
include("../examples/fn_soliton_antisoliton_breather.jl")
include("../examples/fn_soliton_soliton_doublets.jl")

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

# create an instance of the reduced model as specified in fn_reduced_system.jl
function get_reduced_model(T::DataType, encoder, decoder, n, ν_val, N, num_time_points, Ω, time_int, sG::SineGordon; integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    a = T(first(Ω))
    b = T(last(Ω))
    t0 = T(first(time_int))
    t1 = T(last(time_int))
    ν =T(ν_val) 
    params = (ν=ν, N=N, h_N=T((b-a)/(N+1)), φ=(t -> sG.φ(t,ν_val,a)), ψ=(t -> sG.ψ(t,ν_val,b)))
    tstep = T((t1-t0)/(num_time_points-1))
    tspan = (t0, t1)
    ics = vcat(T.(sG.u₀.(Ω,ν)), T.(sG.u₁.(Ω,ν)))
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field_LINEAR(v_field_explicit(params), decoder, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, tspan, tstep, ics; integrator=integrator, system_type=system_type)
end

# create an instance of the reduced model as specified in fn_reduced_system.jl
function get_reduced_model_normalized(T::DataType, encoder, decoder, n, ν_val, N, num_time_points, Ω, time_int, sG::SineGordon; integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    a = T(first(Ω))
    b = T(last(Ω))
    t0 = T(first(time_int))
    t1 = T(last(time_int))
    ν =T(ν_val) 
    params = (ν=ν, N=N, h_N=T((b-a)/(N+1)), φ=(t -> sG.φ(t,ν_val,a)), ψ=(t -> sG.ψ(t,ν_val,b)))
    tstep = T((t1-t0)/(num_time_points-1))
    tspan = (t0, t1)
    ics = vcat(T.(sG.u₀.(Ω,ν)), T.(sG.u₁.(Ω,ν)))
    # CHANGE fn: added field x_ref
    zero = zeros(eltype(ics), size(ics))
    x_ref = ics - decoder(encoder(zero))
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field_normalized_LINEAR(v_field_explicit(params), decoder, x_ref, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, tspan, tstep, ics; x_ref=x_ref, integrator=integrator, system_type=system_type)
end

# generate exact solution of specified data type
function generate_exact_solution(T::DataType, ν_val, Ω, time_int, sG::SineGordon; transformed=false)
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
    if transformed
        return transform(T, Ω[2] - Ω[1], sol_matrix)
    else
        return sol_matrix
    end
end

function get_psd_encoder_decoder(; n=5)
    U = svd(hcat(data[1:N,:], data[(N+1):2*N,:])).U
    Φ = U[:,1:n]
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

T = T_testing_global
N = N_global
sys_dim = 2*N
n_range = range_n_global
num_time_points = time_steps_global + 1
ν_testing_range = T.(ν_testing_range_global)  

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
 
if PSDnormalized_global 
    name_snapshot = "snapshots_normalized_N=$(N_global)"
    name_muerrs = "PSD_errors_normalized_N=$(N_global)"
else
    name_snapshot = "snapshots_N=$(N_global)"
    name_muerrs = "PSD_errors_N=$(N_global)"
end

if TEST_global
    ν_testing_range = T(ν_testing_TEST_global)  
    name_muerrs *= "_TEST"
else
    ν_testing_range = T.(ν_testing_range_global)  
    local n_range_string = "_n="
    for n in range_n_global
        n_range_string = n_range_string * "_$n"
    end
    name_muerrs *= "$n_range_string"
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
raw_time=h5open("resources/sineGordon/data/$(example)_$(name_snapshot).h5", "r") do file
    read(file,"time")
end

# adapt the data type and reshape omega: cut out edges as they are not part of the disrectized hamiltonian, 
# NOTE: this is different from the the initial wave equation example where the hamiltonian dimension was (2N + 2), here we have 2N!!!
omega = T.(raw_omega[2:lastindex(raw_omega) - 1])
time = T.(raw_time)
num_νs = size(raw_data)[3]

# Assert if the globals match with the data loaded
@assert N == length(omega)
@assert n_range == range_n_global
@assert num_time_points == length(time)
@assert sG.T_analytic == eltype(raw_data)

# reshape data; IMPORTANT: cut out q_0, q_{N+1},p_0, p_{N+1} as learned system is 
# of size 2N NOT 2(N+2) other than before with the wave equation!
data = zeros(2*N,num_time_points*num_νs)
for i in 0:num_νs-1
    start_ind = i*num_time_points + 1
    end_ind = (i+1)*num_time_points
    data[1:N,start_ind:end_ind] = raw_data[2:N+1,:,i+1]
    data[N+1:2*N,start_ind:end_ind] = raw_data[N+4:2*(N+2)-1,:,i+1]
end

#endregion


#region Section 5: Code execution

if PSDnormalized_global
    println("Calculating the NORMALIZED errors for example $(example)...")
else
    println("Calculating the UNNORMALIZED errors for example $(example)...")
end

encoders_decoders = get_enocders_decoders(n_range)

ν_errors = NamedTuple()
for ν_test_val in ν_testing_range
    println("   I am at ν=$ν_test_val")
    sol_matrix_full = generate_exact_solution(T, ν_test_val, omega, time, sG)
    errors = NamedTuple()
    for n in n_range
        println("       I am at reduced dimension n=$n")
        current_n_identifier = Symbol("n"*string(n))
        psd_encoder, psd_decoder = encoders_decoders[current_n_identifier]
        if normalized_global
            psd_rs = get_reduced_model_normalized(T, psd_encoder, psd_decoder, n, ν_test_val, N, num_time_points, omega, time, sG)
            psd_red, time_needed, sol_matrix_reconst = compute_reduction_error_analytic_normalized(psd_rs, sol_matrix_full; reconst_sol=plot_reconstructed_sol_global)
            psd_proj=compute_projection_error_analytic_normalized(psd_rs, sol_matrix_full)
        else
            psd_rs = get_reduced_model(T, psd_encoder, psd_decoder, n, ν_test_val, N, num_time_points, omega, time, sG)
            psd_red, time_needed, sol_matrix_reconst = compute_reduction_error_analytic(psd_rs, sol_matrix_full; reconst_sol=plot_reconstructed_sol_global)
            psd_proj=compute_projection_error_analytic(psd_rs, sol_matrix_full)
        end
        println("       Reduction error: $(psd_red)")
        println("       Integration time: $(time_needed.time)")
        if plot_reconstructed_sol_global
            plot_object = Plots.surface(omega, time,transpose(sol_matrix_reconst[1:N,:]), xlabel="ξ", ylabel="t", title="Reconstructed solution for Ω=($(a_global),$(b_global)), I=($(start_time_global),$(end_time_global)), N=$(N_global)", titlefontsize=12)
            if normalized_global
                savefig(plot_object, "plots_temp/reconstructed_PSD_normalized_N=$(N_global)_n=$(n)_ν=$(ν_test_val).eps")
            else
                savefig(plot_object, "plots_temp/reconstructed_PSD_N=$(N_global)_n=$(n)_ν=$(ν_test_val).eps")
            end
        end
        temp_errors = (psd_red, psd_proj)
        errors = NamedTuple{(keys(errors)..., current_n_identifier)}((values(errors)..., temp_errors))
    end
    global ν_errors = NamedTuple{(keys(ν_errors)..., Symbol("ν"*string(ν_test_val)))}((values(ν_errors)..., errors))
end

ν_len =length(ν_testing_range)
n_len = length(n_range) 
error_mat = zeros(ν_len, n_len, 2)
for (ν_ind,ν_key) in zip(1:ν_len,keys(ν_errors))
    for (n_ind, n_key) in zip(1:n_len,keys(ν_errors[ν_key]))
        error_mat[ν_ind,n_ind,:] .= ν_errors[ν_key][n_key]
    end
end

h5open("resources/sineGordon/data/$(example)_$(name_muerrs).h5", "w") do h5
    h5["psd_errors"] = error_mat
    h5["nu_test_range"] = collect(ν_testing_range)
    h5["n_range"] = collect(n_range)
end 

#endregion