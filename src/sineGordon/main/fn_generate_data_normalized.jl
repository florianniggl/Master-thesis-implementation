# ---------------------------------------------------------------------------------------------------------------------
# fn_generate_data_normalized.jl is a SELF-CREATED file to generate normalized data from unnormalized data
# ---------------------------------------------------------------------------------------------------------------------



println("1D SINE GORDON EQUATION: GENERATE NORMALIZED DATA")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using HDF5

include("../general/fn_global_consts.jl")

#endregion


#region Section 2: Initialize parameters 

println("Initializing parameters...")

N = N_global

#endregion


#region Section 3: Load all data required

println("Loading data...")

# example1
data_example1=h5open("resources/sineGordon/data/example1_snapshots_N=$(N).h5", "r") do file
    read(file,"data")
end
omega_example1=h5open("resources/sineGordon/data/example1_snapshots_N=$(N).h5", "r") do file
    read(file,"omega")
end
time_example1=h5open("resources/sineGordon/data/example1_snapshots_N=$(N).h5", "r") do file
    read(file,"time")
end
nu_range_example1=h5open("resources/sineGordon/data/example1_snapshots_N=$(N).h5", "r") do file
    read(file,"nu_range")
end

@assert length(omega_example1) == N_global + 2
@assert length(time_example1) == time_steps_global + 1
@assert length(nu_range_example1) == length(ν_global_range_example1)

# single_soliton
data_single_soliton=h5open("resources/sineGordon/data/single_soliton_snapshots_N=$(N).h5", "r") do file
    read(file,"data")
end
omega_single_soliton=h5open("resources/sineGordon/data/single_soliton_snapshots_N=$(N).h5", "r") do file
    read(file,"omega")
end
time_single_soliton=h5open("resources/sineGordon/data/single_soliton_snapshots_N=$(N).h5", "r") do file
    read(file,"time")
end
nu_range_single_soliton=h5open("resources/sineGordon/data/single_soliton_snapshots_N=$(N).h5", "r") do file
    read(file,"nu_range")
end

@assert length(omega_single_soliton) == N_global + 2
@assert length(time_single_soliton) == time_steps_global + 1
@assert length(nu_range_single_soliton) == length(ν_global_range_single_soliton)

# soliton_antisoliton_breather
data_soliton_antisoliton_breather=h5open("resources/sineGordon/data/soliton_antisoliton_breather_snapshots_N=$(N).h5", "r") do file
    read(file,"data")
end
omega_soliton_antisoliton_breather=h5open("resources/sineGordon/data/soliton_antisoliton_breather_snapshots_N=$(N).h5", "r") do file
    read(file,"omega")
end
time_soliton_antisoliton_breather=h5open("resources/sineGordon/data/soliton_antisoliton_breather_snapshots_N=$(N).h5", "r") do file
    read(file,"time")
end
nu_range_soliton_antisoliton_breather=h5open("resources/sineGordon/data/soliton_antisoliton_breather_snapshots_N=$(N).h5", "r") do file
    read(file,"nu_range")
end

@assert length(omega_soliton_antisoliton_breather) == N_global + 2
@assert length(time_soliton_antisoliton_breather) == time_steps_global + 1
@assert length(nu_range_soliton_antisoliton_breather) == length(ν_global_range_soltion_antisoliton_breather)

# soliton_soliton_doublets
data_soliton_soliton_doublets=h5open("resources/sineGordon/data/soliton_soliton_doublets_snapshots_N=$(N).h5", "r") do file
    read(file,"data")
end
omega_soliton_soliton_doublets=h5open("resources/sineGordon/data/soliton_soliton_doublets_snapshots_N=$(N).h5", "r") do file
    read(file,"omega")
end
time_soliton_soliton_doublets=h5open("resources/sineGordon/data/soliton_soliton_doublets_snapshots_N=$(N).h5", "r") do file
    read(file,"time")
end
nu_range_soliton_soliton_doublets=h5open("resources/sineGordon/data/soliton_soliton_doublets_snapshots_N=$(N).h5", "r") do file
    read(file,"nu_range")
end

@assert length(omega_soliton_soliton_doublets) == N_global + 2
@assert length(time_soliton_soliton_doublets) == time_steps_global + 1
@assert length(nu_range_soliton_soliton_doublets) == length(ν_global_range_soltion_antisoliton_breather)

#endregion


#region Section 4: Code execution

println("Code execution...")

function normalize(A::Array{T,3}) where T <: Real
    data_size = size(A)
    dimension, time_steps, num_νs = data_size  
    data_normalizer = zeros(T, data_size)
    for i in 1:num_νs
        data_normalizer[:,:,i] = hcat(fill(A[:,1,i],time_steps)...)
    end
    A - data_normalizer
end

# create and write example1
h5open("resources/sineGordon/data/example1_snapshots_normalized_N=$(N_global).h5", "w") do h5
    h5["data"] = normalize(data_example1)
    h5["omega"] = omega_example1
    h5["time"] = time_example1
    h5["nu_range"] = nu_range_example1
end

# create and write single_soliton
h5open("resources/sineGordon/data/single_soliton_snapshots_normalized_N=$(N_global).h5", "w") do h5
    h5["data"] = normalize(data_single_soliton)
    h5["omega"] = omega_single_soliton
    h5["time"] = time_single_soliton
    h5["nu_range"] = nu_range_single_soliton
end

# create and write soliton_antisoliton_breather
h5open("resources/sineGordon/data/soliton_antisoliton_breather_snapshots_normalized_N=$(N_global).h5", "w") do h5
    h5["data"] = normalize(data_soliton_antisoliton_breather)
    h5["omega"] = omega_soliton_antisoliton_breather
    h5["time"] = time_soliton_antisoliton_breather
    h5["nu_range"] = nu_range_soliton_antisoliton_breather
end

# create and write soliton_soliton_doublets
h5open("resources/sineGordon/data/soliton_soliton_doublets_snapshots_normalized_N=$(N_global).h5", "w") do h5
    h5["data"] = normalize(data_soliton_soliton_doublets)
    h5["omega"] = omega_soliton_soliton_doublets
    h5["time"] = time_soliton_soliton_doublets
    h5["nu_range"] = nu_range_soliton_soliton_doublets
end

#endregion


