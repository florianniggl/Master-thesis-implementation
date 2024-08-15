# ---------------------------------------------------------------------------------------------------------------------
# fn_global_consts.jl is a SELF-CREATED file to centralize the system setup
#
# Usage:
# 1. set global global_consts
# 2. generate data with fn_generate_analytic_data.jl 
# 3. generate normalized data in fn_generate_data_normalized.jl
# 4. precalculate the PSD-errors in fn_generate_psd (normalized_global decides if normalized or not)
# 5. perform the training (and testing)
# ---------------------------------------------------------------------------------------------------------------------
 


### Variant specific configurations ###
# ---------------------------------------------------------------------------------------------------------------------
# Choice of loss function in use:
const global LossGML_global = true
const global LossModified_global = false
# training setup:
# epochwise_global does not need to be set as we set this to true by default 
const global normalized_global = false
# ---------------------------------------------------------------------------------------------------------------------



### 1D sine-Gordon equation specific parameters ####
# ---------------------------------------------------------------------------------------------------------------------
# full and reduced dimensions
const global N_global = 128 
const global range_n_global = 3:1:15
# number epochs and batch size
const global n_epochs_global = 50
const global batch_size_global = 32
# time span I=[start_time_global, end_time_global and spatial domain Ω=[a,b] 
const global start_time_global = 0.0
#const global end_time_global = 4.0 
const global end_time_global = 3.0 
#const global a_global = -10.0
const global a_global = -3.0
#const global b_global = 10.0 
const global b_global = 3.0
# time steps
const global time_steps_global = 200
# example in use
const global example_in_use_global = "single_soliton"
#const global example_in_use_global = "soliton_soliton_doublets"
# ν values
const global ν_global_range_single_soliton = range(-0.98,-0.72,20) # analogous to wave example generate data for 20 choices of system parameter ν (μ)
const global ν_global_range_soliton_soliton_doublets = range(0.72, 0.98, 20)
const global ν_testing_range_global = [-0.97, -0.92, -0.85, -0.73] #current: single soliton (for soliton_soliton_doublets: positive!!!)
#const global ν_testing_range_global = [0.73, 0.85, 0.92, 0.97] 
# ---------------------------------------------------------------------------------------------------------------------



### Other paramters for customization (do not need to be changed for different training routines)###
# ---------------------------------------------------------------------------------------------------------------------
# option to plot reconstructed curves 
const global plot_reconstructed_sol_global = false;
# option to plot the analytic solution graph
const global analytic_plots_global = false;
# option to save the average epoch loss value (only works with epoch-wise learning)
const global save_epoch_loss_global = false;
# specifying the datatype in use
const global T_analytic_global = Float32 
const global T_testing_global = Float64 # should be left at Float64 as GeometricIntegrators only works with Float64! (CHANGING THIS TYPE CAUSES ERRORS!)
# for development purposes: test simplification values (usually not needed)
const global ν_testing_TEST_global = 0.0  
const global TEST_global = false
const global ν_global = 0.5
# PSD error normalization option:
const global PSDnormalized_global = false # should always be false since we do not want to change the comparison method!
# Other examples tested:
const global ν_global_range_example1 = 0.0:0.5:0.0
const global ν_global_range_soltion_antisoliton_breather = range(0.72, 0.98, 20)
#const global example_in_use_global = "example1"
#const global example_in_use_global = "soliton_antisoliton_breather"
# ---------------------------------------------------------------------------------------------------------------------
