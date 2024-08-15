# ---------------------------------------------------------------------------------------------------------------------
# fn_benchmark_test_homogeneous_optimizer_step.jl is a SELF-CREATED file to measure the performance of one optimizer step
# ---------------------------------------------------------------------------------------------------------------------



println("SPEED TEST: HOMOGENEOUS SPACE OPTIMIZER")
println()


#region Section 1: Load all modules and packages required

println("Loading modules...")

using LinearAlgebra
using GeometricMachineLearning 
using ProgressMeter
using Zygote
using HDF5
using CUDA
using GeometricIntegrators
using Plots
using BenchmarkTools

include("../general/fn_global_consts_alternative_optimizer_specific.jl")

#endregion


#region Section 2: Helper functions...

function print_benchmark(trial::BenchmarkTools.Trial)
    println("Memory estimate: ", trial.memory, " bytes")
    println("Allocations estimate: ", trial.allocs)
    println("Minimum time: ", minimum(trial).time / 1000000, " ms")
    println("Median time: ", median(trial).time / 1000000, " ms")
    println("Mean time: ", mean(trial).time / 1000000, " ms ± ", std(trial).time / 1000000, " ms")
    println("Maximum time: ", maximum(trial).time / 1000000, " ms")
    println("Samples: ", trial.params.samples)
    println("Evals per sample: ", trial.params.evals)
end

#endregion


#region Section 3: Initialize parameters...

println("Initializing parameters...")

pythonplot()

backend = 
try 
    CUDABackend()
catch
    CPU()
end

opt = AdamOptimizer(Float32.((0.001, 0.9, 0.99, 1e-8))...)
retraction = Cayley()

N = N_speed_test_global
n = n_speed_test_global

TESTPSDLayer = PSDLayer(2*N, 2*n; retraction=retraction)
 
ps = initialparameters(backend, Float32, TESTPSDLayer)
optimizer_instance = Optimizer(opt, ps)

test = CUDA.ones(N,n)
dx = NamedTuple{(:weight,)}((test,))

#endregion


#region Section 4: Code execution

println()
println("Calculating step for:")
println("   - optimizer: HomogeneousSpaceAdamOptimizer")
println("   - N: $N")
println("   - n: $n")
println()

# one preiteration as the first iteration always takes longer than a normal intermediate iteration
optimization_step!(optimizer_instance, TESTPSDLayer, ps, optimizer_instance.cache, dx)

println("Calculating step...")
timer = @benchmark optimization_step!(optimizer_instance, TESTPSDLayer, ps, optimizer_instance.cache, dx)
println("Results:")
print_benchmark(timer)

#endregion