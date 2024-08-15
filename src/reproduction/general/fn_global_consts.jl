# ---------------------------------------------------------------------------------------------------------------------
# fn_global_consts.jl is a SELF-CREATED file to centralize the system setup
#
# Usage:
# 1. set global global_consts
# 2. generate data with fn_integration_v2.jl 
# 3. generate normalized data in fn_generate_data_normalized.jl
# 4. precalculate the PSD-errors in fn_generate_psd.jl
# 5. perform the training (and testing)
# ---------------------------------------------------------------------------------------------------------------------



### Variant specific configurations ###
# ---------------------------------------------------------------------------------------------------------------------
# Choice of loss function in use:
const global LossGML_global = true
const global LossModified_global = false
# training setup:
const global epochwise_global = true
const global normalized_global = true
# ---------------------------------------------------------------------------------------------------------------------



### 1D wave equation specific parameters ####
#Note: time span and Ω cannot be chosen freely, they  are set to I×Ω = (0,1)×(-1/2,1/2))
# ---------------------------------------------------------------------------------------------------------------------
# full and reduced dimensions
const global N_global = 128 
const global n_range_global = 2:1:15
# number epochs and batch size
const global n_epochs_global = 50 
const global batch_size_global = 32
# time steps
const global time_steps_global = 200 
# μ values
const global n_params_global = 20
const global μ_left_global = 5/12 
const global μ_right_global = 2/3 
const global μ_testing_range_global = (0.47, 0.51, 0.55, 0.625)  
# ---------------------------------------------------------------------------------------------------------------------



### Other paramters for customization (do not need to be changed for different training routines)###
# ---------------------------------------------------------------------------------------------------------------------
# option to plot reconstructed curves 
const global plot_reconstructed_sol_global = false;
# option to save the average epoch loss value (only works with epoch-wise learning)
const global save_epoch_loss_global = false;
# set headline and y-axis range for the training output error plots
const global headline_error_plots_global = "Errors for Ω=(-0.5,0.5), I=(0,1), N=$N_global"
const global y_limits_global = 1
# specifying the datatype in use
const global T_learning_global = Float64 #should be set to Float64 for training data integration, for learing Float32 can be used 
const global T_testing_global = Float64 # should be left at Float64 as GeometricIntegrators only works with Float64! (CHANGING THIS TYPE CAUSES ERRORS!)
# choice of initial condition for p (should always be set to false since we always tested on the homogenous Dirichlet condition)
const global p_zero_global = false
# for development purposes: test simplification values (usually not needed)
const global μ_testing_TEST_global = 0.51  
const global TEST_global = false
# ---------------------------------------------------------------------------------------------------------------------