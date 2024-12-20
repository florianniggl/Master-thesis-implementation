# ---------------------------------------------------------------------------------------------------------------------
# fn_fn_plot_waves_v2.jl is a MODIFICATION of plot_waves.jl from GeometricMachineLearning@v0.2.0
# Note: the original file is not part of the source code of GeometricMachineLearning.jl release package, instead it is 
# part of test setups in scripts/symplectic_autoencoders
# ---------------------------------------------------------------------------------------------------------------------



println("1D WAVE EQUATION: PLOT WAVES")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using HDF5, Plots

include("../general/fn_global_consts.jl")

#endregion


#region Section 2: Create wave plot from snapshot matrix

println("Create plots...")

if p_zero_global
    name = "snapshot_matrix_pzero_N=$(N_global)"
else
    name = "snapshot_matrix_N=$(N_global)"
end


data = h5open("resources/reproduction/$(name).h5", "r") do file
    read(file, "data")
end
n_params = h5open("resources/reproduction/$(name).h5", "r") do file
    read(file, "n_params")
end

function indices(number_total_indices::Int, number_indices::Int)
    spacing = number_total_indices/number_indices
    Int.(ceil.(1:spacing:number_total_indices))
end

μ_left = μ_left_global 
μ_right = μ_right_global
μ_collection = μ_left:((μ_right - μ_left) / (n_params - 1)):μ_right 

function plot_curves(number_of_params::Int=3, time_instances::Int=3)
    number_time_indices = size(data, 2)÷n_params

    param_indices = indices(n_params, number_of_params)
    time_indices = indices(number_time_indices, time_instances)

    time_labels = reshape((0:(1 / (number_time_indices - 1)):1)[time_indices], 1, time_instances)

    N = size(data,1)÷2

    Ω = -0.5:(1 / (N - 1)):0.5

    plot_object = plot(; layout=(time_instances,1))
    index_number = 0

    function title_gen(t::Real)
        output = try 
            "t="*string(t)[1:4]
        catch 
            "t="*string(t)
        end
        output
    end

    for param_index in param_indices
        index_number += 1
        data_to_plot = data[1:N, (param_index-1) * number_time_indices .+ time_indices]
        plot!(
            plot_object, 
            Ω, 
            data_to_plot, 
            layout=(time_instances, 1), 
            color=3+index_number, 
            label="μ="*string(μ_collection[param_index])[1:5],
            title=title_gen.(time_labels), 
            titleloc = :left, 
            titlefont = font(12)
            )
    end
    plot_object
end
p1 =  plot_curves(3, 3)
display(p1)
sleep(1000)
#png(plot_curves(3, 3), "plots/wave_plot")

#endregion