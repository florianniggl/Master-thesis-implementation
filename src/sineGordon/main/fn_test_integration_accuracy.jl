# ---------------------------------------------------------------------------------------------------------------------
# fn_test_integration_accuracy.jl is a SELF-CREATED TEST-file to test the integration accuracy of the different sG-examples
# ---------------------------------------------------------------------------------------------------------------------



println("1D SINE GORDON EQUATION: TEST INTEGRATION ACCURACY")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using LinearAlgebra
using GeometricIntegrators
using Plots
using HDF5
using GeometricMachineLearning

using ForwardDiff # needed in sine_gordon.jl
import GeometricMachineLearning: SystemType    #needed in fn_reduced_system_v2.jl

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

# plot and safe the reconstruced curves to file
function plot_reconstruced_curves(sol_matrix, Ω, time_int, a, b, t0, t1, n)
    plot_object = Plots.surface(Ω,time_int,transpose(sol_matrix[1:length(Ω),:]), xlabel="ξ", ylabel="t", title="Reconstructed solution for Ω=($(a),$(b)), I=($(t0),$(t1)), n=$(n)", titlefontsize=12)
    png(plot_object, "plots_temp/$(example)_n=$(n)")
end

# plot and safe the exact curves to file
function plot_exact_curves(sol_matrix, Ω, time_int, a, b, t0, t1, n)
    plot_object = Plots.surface(Ω,time_int,transpose(sol_matrix[1:length(Ω),:]), xlabel="ξ", ylabel="t", title="Exact solution for Ω=($(a),$(b)), I=($(t0),$(t1)), n=$(n)", titlefontsize=12)
    png(plot_object, "plots_temp/$(example)_exact_n=$(n)")
end

# state-space transformation of specified data type
function transform(T::DataType, h_N, A::AbstractArray)
    T.(sqrt(h_N) * A)
end

# state-space back-transformation of specified data type
function back_transform(T::DataType, h_N, A::AbstractArray)
    T.(sqrt(1/h_N) * A)
end

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

function get_psd_encoder_decoder(; n=5)
    Φ = svd(hcat(I(n), I(n))).U[:,1:n]
    println(Φ)
    PSD = hcat(vcat(Φ, zero(Φ)), vcat(zero(Φ), Φ))

    PSD_cpu = _cpu_convert(PSD)
    psd_encoder(z) = PSD_cpu'*z 
    psd_decoder(ξ) = PSD_cpu*ξ
    psd_encoder, psd_decoder
end

#endregion


#region Section 3: Initializing parameters

println("Initializing parameters...")

# specify sG example
#example = "example1"
example = "single_soliton"
#example = "soliton_antisoliton_breather"
#example = "soliton_soliton_doublets"

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

transformed = false
reduced_field_calculation = true
psd_decoder = false # if false, uses "optimal" function Identity!, but gives integration warning, but which should not affect "redundant" calculation

n_range = 15:1:15
#_range = 128:10:128
#n_range = 10:20:50
#n_range = 500:2:500
#n_range = 3:2:23
#n_range = 10:20:70

sG = sG_constructor(
   -10.0,
   10.0,
   0.0,
   4.0,
   0.3:0.5:0.3;
   #0.0:0.5:0.0;
   T_analytic=T_testing_global,
   N=nothing,
   #t_steps=399)
   t_steps=199)

#endregion


#region Section 4: Code execution

println("Calculating the error...")

error_vals = ()
int_times = ()
for n in n_range 
    h_n = sG.T_analytic((sG.b-sG.a)/(n+1))
    ν = sG.ν_range[1]
    params = (ν=ν, N=n, h_N=h_n, φ=(t -> sG.φ(t,ν,sG.a)), ψ=(t -> sG.ψ(t,ν,sG.b)))
    tstep = sG.T_analytic((sG.t1-sG.t0)/(sG.t_steps))
    tspan = (sG.t0, sG.t1)
    omega = (sG.a:h_n:sG.b)[2:n+1]
    time = sG.t0:tstep:sG.t1
    
    if psd_decoder 
        decoder = get_psd_encoder_decoder(n=n_range[1])[2]
    else
        Id(x) =  x
        decoder = Id 
    end
    
    if transformed 
        ics = transform(sG.T_analytic,h_n, vcat(sG.T_analytic.(sG.u₀.(omega,ν)), sG.T_analytic.(sG.u₁.(omega,ν))))
        #plot_object = Plots.plot(omega, time, ics)
        #display(plot_object)
        #sleep(1000)
        if reduced_field_calculation
            v_field_full = reduced_vector_field_from_full_explicit_vector_field(v_field_transformed_explicit(params), decoder, 2*n, 2*n)
        else
            v_field_full = v_field_transformed(params)
        end
    else
        ics = vcat(sG.T_analytic.(sG.u₀.(omega,ν)), sG.T_analytic.(sG.u₁.(omega,ν)))
        if reduced_field_calculation
            v_field_full = reduced_vector_field_from_full_explicit_vector_field(v_field_explicit(params), decoder, n, n)
        else
            v_field_full = v_field(params)
        end
    end

    sol_matrix_analytic = generate_exact_solution(sG.T_analytic, ν, omega, time, sG)
    
    ode = ODEProblem(v_field_full, parameters=params, tspan, tstep, ics)

    int_time =
    @timed begin
    sol_integrated = integrate(ode, ImplicitMidpoint())
    end

    sol_matrix_integrated = zeros(2*n, sG.t_steps + 1)
    for (t_ind,q) in zip(1:sG.t_steps + 1,sol_integrated.q)
        sol_matrix_integrated[:, t_ind] = q
    end
    if transformed
        sol_matrix_integrated .= back_transform(sG.T_analytic, h_n, sol_matrix_integrated)
    end
    
    #plot_reconstruced_curves(sol_matrix_integrated, omega, time, first(omega), last(omega), first(time), last(time), n)
    #plot_exact_curves(sol_matrix_analytic, omega, time, first(omega), last(omega), first(time), last(time), n)
    
    error_red = norm(sol_matrix_integrated - sol_matrix_analytic)/norm(sol_matrix_analytic)
    global error_vals = (error_vals...,error_red)
    global int_times = (int_times...,int_time)
end

println("Writing the log data...")
open("plots_temp/integration_accuracy_$example.txt", "w") do file

    write(file, "Parameter setup: \n\n")

    write(file, "Example: $example \n")
    write(file, "Dimension range: $(n_range) \n")
    write(file, "Time steps: $(sG.t_steps) \n")
    write(file, "Domain: Ω=($(sG.a), $(sG.b)), I=($(sG.t0), $(sG.t1)) \n")
    write(file, "ν_val: $(sG.ν_range[1]) \n")
    write(file, "Transformed: $(transformed) \n")
    write(file, "Reduced field form: $(reduced_field_calculation) \n")
    write(file, "PSD Decoder: $(psd_decoder) \n")
    write(file, "DataType used: $(sG.T_analytic) \n\n")

    write(file, "Calculated integration errors: \n\n")
    for i in eachindex(n_range)
        write(file, "Relative error in dimension n=$(n_range[i]): $(error_vals[i]) \n")
        write(file, "Integration time needed in dimension n=$(n_range[i]): $(int_times[i].time) \n\n")
    end
end

#plot1 = Plots.surface(omega,time,transpose(sol_matrix_analytic[1:length(omega),:]), xlabel="ξ", ylabel="t", title="exact_solution")
#plot2 = Plots.surface(omega,time,transpose(sol_matrix_integrated[1:length(omega),:]), xlabel="ξ", ylabel="t", title="reconstructed_solution")
#plot = Plots.plot(plot1, plot2, layout=(1,2))
#display(plot)
#sleep(1000)

#endregion


