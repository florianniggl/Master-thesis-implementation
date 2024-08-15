# ---------------------------------------------------------------------------------------------------------------------
# fn_global_consts_alternative_optimizer_specific.jl is a SELF-CREATED EXTENSION-file to extend fn_global_consts.jl with 
# optimizer specific data:
#   - choose (alternative optimizer) to be used
#   - choose metric to be used to calculate the Riemannian gradient 
#   - choose vector transport to be used for optimization step
# ---------------------------------------------------------------------------------------------------------------------



### Variant specific configurations ###
# ---------------------------------------------------------------------------------------------------------------------
# optimizer instance:
const global StiefelAdamOptimizer_global = false   
const global StiefelAdamOptimizerWithDecay_global = true   
# Choice of metric in use:
const global CanonicalMetric_global = true
const global EuclideanMetric_global = false
# Choice of vector transport in use:
const global SubmanifoldVectorTransport_global = true
const global DifferentialVectorTransport_global = false
# ---------------------------------------------------------------------------------------------------------------------



### Other optimizer options ###
# ---------------------------------------------------------------------------------------------------------------------
const global StiefelStochasticGradientDescentWithMomentum_global = false    
const global StiefelStochasticGradientDescentWithMomentumWithDecay_global = false    
const global StiefelCayleyAdamOptimizerFromOtherPaper_global = false    
# ---------------------------------------------------------------------------------------------------------------------



### Benchmark test specific configurations ###
# only needed for fn_benchmark_test_alternative_optimizer.jl and fn_benchmark_test_homogeneous_optimizer.jl
# ---------------------------------------------------------------------------------------------------------------------
const global N_speed_test_global = 10000
const global n_speed_test_global = 10
# ---------------------------------------------------------------------------------------------------------------------