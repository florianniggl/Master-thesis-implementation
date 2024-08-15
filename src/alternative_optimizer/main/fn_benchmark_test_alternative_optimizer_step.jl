# ---------------------------------------------------------------------------------------------------------------------
# fn_benchmark_test_alternative_optimizer_step.jl is a SELF-CREATED file to measure the performance of one optimizer step
# ---------------------------------------------------------------------------------------------------------------------



println("SPEED TEST: ALTERNATIVE OPTIMIZER")
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
using KernelAbstractions
using BenchmarkTools

using Random    #needed in fn_batch_v2.jl
import GeometricMachineLearning: LayerWithManifold  #needed in fn_retractions_v2.jl
using AbstractNeuralNetworks: AbstractExplicitLayer, AbstractExplicitCell   #needed in fn_retractions_v2.jl

pythonplot()

include("../../shared/fn_customtypes.jl")

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

#include("../../shared/fn_GML_custom/fn_batch_v2.jl")

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

backend = 
try 
    CUDABackend()
catch
    CPU()
end

T = Float32

if StiefelAdamOptimizer_global
    opt = StiefelAdamOptimizer(T.((0.001, 0.9, 0.99, 1e-8))...)
    opt_name = "StiefelAdamOptimizer"
elseif StiefelAdamOptimizerWithDecay_global
    opt = StiefelAdamOptimizerWithDecay(T.((0.001, 0.9, 0.9, 0.99, 0.99, 1e-8))...) 
    opt_name = "StiefelAdamOptimizerWithDecay"
elseif StiefelStochasticGradientDescentWithMomentum_global
    opt = StiefelStochasticGradientDescentWithMomentum(T.((0.001, 0.9))...)
    opt_name = "StiefelStochasticGradientDescentWithMomentum"
elseif StiefelStochasticGradientDescentWithMomentumWithDecay_global
    opt = StiefelStochasticGradientDescentWithMomentumWithDecay(T.((0.001, 0.9))...)
    opt_name = "StiefelStochasticGradientDescentWithMomentumWithDecay"
elseif StiefelCayleyAdamOptimizerFromOtherPaper_global
    opt = StiefelCayleyAdamOptimizerFromOtherPaper(T.((0.4, 0.9, 0.9, 0.99, 0.99, 1e-8, 1.0, 0.5, 2))...)
    opt_name = "StiefelCayleyAdamOptimizerFromOtherPaper"
else
    # if non is selected, throw optimizer not selected exception
    throw(NoOptimizerSelected("An optimizer has to  be selected."))
end

if CanonicalMetric_global
    met = Canonical()
elseif EuclideanMetric_global
    met = Euclidean()
else
    throw(GlobalConstantError("A metric has to be chosen."))
end

if SubmanifoldVectorTransport_global
    vt = SubmanifoldVectortransport()
elseif DifferentialVectorTransport_global
    vt = DifferentialVectortransport()
else
    throw(GlobalConstantError("A vector transport has to be chosen."))
end

retraction = Cayley()

N = N_speed_test_global
n = n_speed_test_global

TESTPSDLayer = PSDLayer(2*N, 2*n; retraction=retraction)
 
ps = initialparameters(backend, T, TESTPSDLayer)
optimizer_instance = Optimizer(opt, ps)

test = CUDA.ones(N,n)
dx = NamedTuple{(:weight,)}((test,))

#endregion


#region Section 4: Code execution

println()
println("Calculating step for:")
println("   - optimizer: $opt_name")
println("   - N: $N")
println("   - n: $n")
println()

# one preiteration as the first iteration always takes longer than a normal intermediate iteration
optimization_step!(optimizer_instance, met, vt, TESTPSDLayer, ps, optimizer_instance.cache, dx)

println("Calculating step...")
timer = @benchmark optimization_step!(optimizer_instance, met, vt, TESTPSDLayer, ps, optimizer_instance.cache, dx)
println("Results:")
print_benchmark(timer)

#endregion





