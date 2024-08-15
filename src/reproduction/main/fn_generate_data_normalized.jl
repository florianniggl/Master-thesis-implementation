# ---------------------------------------------------------------------------------------------------------------------
# fn_generate_data_normalized.jl is a SELF-CREATED file to generate normalized data from unnormalized data
# ---------------------------------------------------------------------------------------------------------------------



println("1D WAVE EQUATION: GENERATE NORMALIZED DATA")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using HDF5

include("../general/fn_global_consts.jl")

#endregion


#region Section 2: Initialize parameters 

println("Initializing parameters...")

T = T_learning_global
n_time_steps = time_steps_global

if p_zero_global
    name = "snapshot_matrix_pzero_N=$(N_global)"
    name_norm = "snapshot_matrix_normalized_pzero_N=$(N_global)"
else
    name = "snapshot_matrix_N=$(N_global)"
    name_norm = "snapshot_matrix_normalized_N=$(N_global)"
end

#endregion


#region Section 4: Load all data required

println("Loading data...")

data = h5open("resources/reproduction/$(name).h5", "r") do file
        read(file, "data")
    end
n_params = h5open("resources/reproduction/$(name).h5", "r") do file
        read(file, "n_params")
    end

@assert n_time_steps == Int(size(data,2) / n_params)

#endregion


#region Section 5: Code execution

println("Code execution...")

data_normalizer = zeros(eltype(data), size(data))

for i in 0:(n_params-1)
    ind_start = i*n_time_steps + 1
    ind_end = i*n_time_steps + n_time_steps
    data_normalizer[:, ind_start:ind_end] = hcat(fill(data[:, ind_start], ind_end - ind_start + 1)...)
end

data_normalized = data - data_normalizer

h5open("resources/reproduction/$(name_norm).h5", "w") do h5
    h5["data"] = data_normalized
    h5["n_params"] = n_params 
end

#endregion