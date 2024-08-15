# ---------------------------------------------------------------------------------------------------------------------
# fn_generate_analytic_data.jl is a SELF-CREATED file to generate analytic data snapshot matrix
# ---------------------------------------------------------------------------------------------------------------------



println("1D SINE GORDON EQUATION: ANALYTIC DATA GENERATION")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using Plots, ForwardDiff
using HDF5

pythonplot()

include("../general/fn_global_consts.jl")
include("../general/fn_math_utils.jl")

include("../examples/fn_sine_gordon.jl")
include("../examples/fn_example1.jl")
include("../examples/fn_single_soliton.jl")
include("../examples/fn_soliton_antisoliton_breather.jl")
include("../examples/fn_soliton_soliton_doublets.jl")

#endregion

    
#region Section 2: Helper functions

function generate_data(sG::SineGordon, generate_plots)
    T = sG.T_analytic
    #discretizing Ω into (N+2) points
    spacing = T( (sG.b - sG.a) / (sG.N + 1) )
    Ω = T(sG.a):spacing:T(sG.b)
    #discretizing the time interval into num_time_steps + 1 points
    t_step = T( (sG.t1 - sG.t0) / sG.t_steps )
    time_int = T(sG.t0):t_step:T(sG.t1)
    
    #save data to 3D matrix: (ξ,t,ν)
    curves = zeros(T,2*length(Ω), length(time_int), length(sG.ν_range))
    for it in zip(axes(time_int, 1), time_int)
        i = it[1]; t = it[2]
        for it2 in zip(axes(sG.ν_range, 1), sG.ν_range)
            j = it2[1]; ν = T(it2[2])
            for it3 in zip(axes(Ω,1),Ω)
                k=it3[1]; ξ=it3[2]
                curves[k, i, j] = sG.exact_solution(t,ξ,ν)
                curves[k + (sG.N + 2), i, j] = ForwardDiff.derivative(t -> sG.exact_solution(t,ξ,ν), t)
            end
        end
    end

    omega = collect(Ω) 
    time = collect(time_int)
    nu_range = collect(sG.ν_range)
    if generate_plots
        plots=NamedTuple()
        i=1
        for ν in sG.ν_range
           plot = Plots.surface(Ω,time_int,transpose(curves[1:length(Ω),:,i]), xlabel="ξ", ylabel="t", title="Analytic solution for Ω=($(sG.a),$(sG.b)), I=($(sG.t0),$(sG.t1)), N=$(sG.N)", titlefontsize=12)
           symb=Symbol(string(ν))
           plots=NamedTuple{(keys(plots)...,symb)}((values(plots)...,plot))
           i+=1
        end
        return curves, omega, time, nu_range, plots
    end

    return curves, omega, time, nu_range
end

#endregion


#region Section 3: Perform analytic data generation

println("Code execution...")

# instantiate analytic solution examples
example1 = SineGordon_Example1(a_global, b_global, start_time_global, end_time_global, ν_global_range_example1)
single_soliton = SineGordon_SingleSoliton(a_global, b_global, start_time_global, end_time_global, ν_global_range_single_soliton)
soliton_antisoliton_breather = SineGordon_SolitonAntisolitonBreather(a_global, b_global, start_time_global, end_time_global, ν_global_range_soltion_antisoliton_breather)
soliton_soliton_doublets = SineGordon_SolitonSolitonDoublet(a_global, b_global, start_time_global, end_time_global, ν_global_range_soliton_soliton_doublets)

if analytic_plots_global
    # generate the data
    data_example1, omega_example1, time_example1, nu_range_example1, plots_example1 = generate_data(example1, analytic_plots_global)
    data_single_soliton, omega_single_soliton, time_single_soliton, nu_range_single_soliton, plots_single_soliton = generate_data(single_soliton, analytic_plots_global)
    data_soliton_antisoliton_breather, omega_soliton_antisoliton_breather, time_soliton_antisoliton_breather, nu_range_soliton_antisoliton_breather, plots_soliton_antisoliton_breather = generate_data(soliton_antisoliton_breather, analytic_plots_global)
    data_soliton_soliton_doublets, omega_soliton_soliton_doublets, time_soliton_soliton_doublets, nu_range_soliton_soliton_doublets, plots_soliton_soliton_doublets = generate_data(soliton_soliton_doublets, analytic_plots_global)

    # save plots 
    for (ν, plot) in pairs(plots_example1)
        Plots.savefig(plot, "plots_temp/example1_N=$(N_global)_ν=$ν.eps")
    end
    for (ν, plot) in pairs(plots_single_soliton)
        Plots.savefig(plot, "plots_temp/single_soliton_N=$(N_global)_ν=$ν.eps")
    end
    for (ν, plot) in pairs(plots_soliton_antisoliton_breather)
        Plots.savefig(plot, "plots_temp/soliton_antisoliton_breather_N=$(N_global)_ν=$ν.eps")
    end
    for (ν, plot) in pairs(plots_soliton_soliton_doublets)
        Plots.savefig(plot, "plots_temp/soliton_soliton_doublets_N=$(N_global)_ν=$ν.eps")
    end
else
    # generate the data
    data_example1, omega_example1, time_example1, nu_range_example1 = generate_data(example1, analytic_plots_global)
    data_single_soliton, omega_single_soliton, time_single_soliton, nu_range_single_soliton = generate_data(single_soliton, analytic_plots_global)
    data_soliton_antisoliton_breather, omega_soliton_antisoliton_breather, time_soliton_antisoliton_breather, nu_range_soliton_antisoliton_breather = generate_data(soliton_antisoliton_breather, analytic_plots_global)
    data_soliton_soliton_doublets, omega_soliton_soliton_doublets, time_soliton_soliton_doublets, nu_range_soliton_soliton_doublets = generate_data(soliton_soliton_doublets, analytic_plots_global)
end

# write data to a .h5 file
h5open("resources/sineGordon/data/example1_snapshots_N=$(N_global).h5", "w") do h5
    h5["data"] = data_example1
    h5["omega"] = omega_example1
    h5["time"] = time_example1
    h5["nu_range"] = nu_range_example1
end
h5open("resources/sineGordon/data/single_soliton_snapshots_N=$(N_global).h5", "w") do h5
    h5["data"] = data_single_soliton
    h5["omega"] = omega_single_soliton
    h5["time"] = time_single_soliton
    h5["nu_range"] = nu_range_single_soliton
end
h5open("resources/sineGordon/data/soliton_antisoliton_breather_snapshots_N=$(N_global).h5", "w") do h5
    h5["data"] = data_soliton_antisoliton_breather
    h5["omega"] = omega_soliton_antisoliton_breather
    h5["time"] = time_soliton_antisoliton_breather
    h5["nu_range"] = nu_range_soliton_antisoliton_breather
end
h5open("resources/sineGordon/data/soliton_soliton_doublets_snapshots_N=$(N_global).h5", "w") do h5
    h5["data"] = data_soliton_soliton_doublets
    h5["omega"] = omega_soliton_soliton_doublets
    h5["time"] = time_soliton_soliton_doublets
    h5["nu_range"] = nu_range_soliton_soliton_doublets
end

#endregion