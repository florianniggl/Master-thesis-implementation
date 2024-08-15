# ---------------------------------------------------------------------------------------------------------------------
# fn: fn_reduced_system_v2.jl REPLACES reduced_system.jl from GeometricMachineLearning@v0.2.0 
# ---------------------------------------------------------------------------------------------------------------------


#region Section 1: struct ReducedSystem

# fn CHANGE: add struct:
# to modify existing struct ReducedSystem to work with normalized data 
struct ReducedSystem{T, ST<:SystemType} 
    N::Integer 
    n::Integer
    encoder
    decoder 
    full_vector_field
    reduced_vector_field 
    integrator 
    params
    tspan 
    tstep
    ics
    x_ref   # added field x_ref to work with normalized data

    function ReducedSystem(N::Integer, n::Integer, encoder, decoder, full_vector_field, reduced_vector_field, params, tspan, tstep, ics; x_ref=nothing, integrator=ImplicitMidpoint(), system_type=Symplectic(), T=Float64) 
        new{T, typeof(system_type)}(N, n, encoder, decoder, full_vector_field, reduced_vector_field, integrator, params, tspan, tstep, ics, x_ref)
    end
end

# CHANGE fn: added method: 
# made calculation more efficiently (this inefficiency was REMOVED in v0.3.0)
function reduced_vector_field_from_full_explicit_vector_field(full_explicit_vector_field, decoder, N::Integer, n::Integer)
    function reduced_vector_field(v, t, ξ, params)
        x = full_explicit_vector_field(t, decoder(ξ), params)
        Del = ForwardDiff.jacobian(decoder, ξ)'
        v .=  vcat(-Del[n+1:2*n,:], Del[1:n,:]) * vcat(x[N+1:2*N],-x[1:N])
        # Implementation in v0.2.0:
        # v .= -SymplecticPotential(2*n) * ForwardDiff.jacobian(decoder, ξ)' * SymplecticPotential(2*N) * full_explicit_vector_field(t, decoder(ξ), params)
    end
    reduced_vector_field
end

# CHANGE fn: added method:
# -to work with normalized data setup
# -made calculation more efficiently (this inefficiency was REMOVED in v0.3.0)
function reduced_vector_field_from_full_explicit_vector_field_normalized(full_explicit_vector_field, decoder, x_ref, N::Integer, n::Integer)
    function x_ref_plus_d(ξ)
        return x_ref + decoder(ξ)
    end
    function reduced_vector_field(v, t, ξ, params)
        x = full_explicit_vector_field(t, x_ref_plus_d(ξ), params)
        Del = ForwardDiff.jacobian(decoder, ξ)'
        v .=  vcat(-Del[n+1:2*n,:], Del[1:n,:]) * vcat(x[N+1:2*N],-x[1:N])
        #v .= -SymplecticPotential(n) * ForwardDiff.jacobian(decoder, ξ)' * SymplecticPotential(N) * full_explicit_vector_field(t, x_ref_plus_d(ξ), params)
    end
    reduced_vector_field
end

# CHANGE fn: added method: 
# works with linear decoder function (more efficient implementation!)
# made calculation more efficiently (this inefficiency was REMOVED in v0.3.0)
function reduced_vector_field_from_full_explicit_vector_field_LINEAR(full_explicit_vector_field, decoder, N::Integer, n::Integer)
    if decoder !== nothing
        Del = decoder(I(2*n))'
    end
    function reduced_vector_field(v, t, ξ, params)
        x = full_explicit_vector_field(t, decoder(ξ), params)
        v .=  vcat(-Del[n+1:2*n,:], Del[1:n,:]) * vcat(x[N+1:2*N],-x[1:N])
        # Implementation in v0.2.0:
        # v .= -SymplecticPotential(2*n) * ForwardDiff.jacobian(decoder, ξ)' * SymplecticPotential(2*N) * full_explicit_vector_field(t, decoder(ξ), params)
    end
    reduced_vector_field
end

# CHANGE fn: added method:
# -to work with normalized data setup and linear decoder function (more efficient implementation!)
# -made calculation more efficiently (this inefficiency was REMOVED in v0.3.0)
function reduced_vector_field_from_full_explicit_vector_field_normalized_LINEAR(full_explicit_vector_field, decoder, x_ref, N::Integer, n::Integer)
    function x_ref_plus_d(ξ)
        return x_ref + decoder(ξ)
    end
    if decoder !== nothing
        Del = decoder(I(2*n))'
    end
    function reduced_vector_field(v, t, ξ, params)
        x = full_explicit_vector_field(t, x_ref_plus_d(ξ), params)
        v .=  vcat(-Del[n+1:2*n,:], Del[1:n,:]) * vcat(x[N+1:2*N],-x[1:N])
        #v .= -SymplecticPotential(n) * ForwardDiff.jacobian(decoder, ξ)' * SymplecticPotential(N) * full_explicit_vector_field(t, x_ref_plus_d(ξ), params)
    end
    reduced_vector_field
end

#endregion


#region Section 2: Integration

# CHANGE fn: added method:
# - to obtain time record
# - to perform_integration_reduced to work with UNNORMALIZED DATA setup (functionality the same as in reduced_system.jl)
function perform_integration_reduced(rs::ReducedSystem; get_time=false)
    ics_reduced = rs.encoder(rs.ics)
    ode = ODEProblem(rs.reduced_vector_field, parameters=rs.params, rs.tspan, rs.tstep, ics_reduced)
    int_time =
    @timed begin
        result = integrate(ode, rs.integrator)
    end
    if get_time
        return result, int_time
    else
        return result
    end
end

# CHANGE fn: added method:
# - to obtain time record
# - to perform_integration_reduced_normalized to work with NORMALIZED DATA setup
function perform_integration_reduced_normalized(rs::ReducedSystem; get_time=false)
    ics_reduced = rs.encoder(zeros(eltype(rs.ics), size(rs.ics)))
    ode = ODEProblem(rs.reduced_vector_field, parameters=rs.params, rs.tspan, rs.tstep, ics_reduced)
    int_time =
    @timed begin
        result = integrate(ode, rs.integrator)
    end
    if get_time
        return result, int_time
    else
        return result
    end
end

### perform_integration_full (remains unchanged from reduced_system.jl)

function perform_integration_full(rs::ReducedSystem)
    ode = ODEProblem(rs.full_vector_field, parameters=rs.params, rs.tspan, rs.tstep, rs.ics)
    integrate(ode, rs.integrator)
end

#endregion


#region Section 3: Reduction errors

# CHANGE fn: added method:
# - to compute_reduction_error to work with UNNORMALIZED DATA (like is done in reduced_system.jl)
# - to obtain reconstructed solution
# - to plot reduced solution
# - to obtain time record 
function compute_reduction_error(rs::ReducedSystem, sol_full; reconst_sol=false, plot_red_sol=false)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_red, time_needed = perform_integration_reduced(rs; get_time=true)
    sol_matrix_red = zeros(2*rs.n, n_time_steps)
    sol_matrix_reconst = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_red.q)
        sol_matrix_red[:,t_ind] = q
        sol_matrix_reconst[:, t_ind] = rs.decoder(q)
    end
    sol_matrix_full = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    if plot_red_sol
        plot_object = Plots.surface(1:1:rs.n, 1:1:num_time_steps,transpose(sol_matrix_red[1:rs.n,:]), xlabel="ξ", ylabel="t", title="Reduced solution", titlefontsize=12)
        savefig(plot_object, "plots_temp/reduced_sol_n=$(rs.n).eps")
    end
    if reconst_sol
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, sol_matrix_reconst
    else
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, nothing
    end
end

# CHANGE fn: added method:
# - to compute_reduction_error_normalized to work with NORMALIZED DATA
# - to obtain reconstructed solution
# - to plot reduced solution
# - to obtain time record
function compute_reduction_error_normalized(rs::ReducedSystem, sol_full; reconst_sol=false, plot_red_sol=false)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_red, time_needed = perform_integration_reduced_normalized(rs; get_time=true)
    sol_matrix_red = zeros(2*rs.n, n_time_steps)
    sol_matrix_reconst = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_red.q)
        sol_matrix_red[:,t_ind] = q
        sol_matrix_reconst[:, t_ind] = rs.x_ref + rs.decoder(q)
    end
    sol_matrix_full = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    if plot_red_sol
        plot_object = Plots.surface(1:1:rs.n, 1:1:num_time_steps,transpose(sol_matrix_red[1:rs.n,:]), xlabel="ξ", ylabel="t", title="Reduced solution", titlefontsize=12)
        savefig(plot_object, "plots_temp/reduced_sol_n=$(rs.n).eps")
    end
    if reconst_sol
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, sol_matrix_reconst
    else
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, nothing
    end
end

# CHANGE fn: added method:
# - to compute_reduction_error_analytic to work with ANALYTIC DATA (needed for SG examples)
# - to obtain time record
# - to obtain reconstructed solution
# - to plot reduced solution
function compute_reduction_error_analytic(rs::ReducedSystem, sol_matrix_full; reconst_sol=false, plot_red_sol=false)
    num_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_red, time_needed = perform_integration_reduced(rs; get_time=true)
    sol_matrix_red = zeros(2*rs.n, num_time_steps)
    sol_matrix_reconst = zeros(2*rs.N, num_time_steps)
    for (t_ind,q) in zip(1:num_time_steps,sol_red.q)
        sol_matrix_red[:,t_ind] = q
        sol_matrix_reconst[:, t_ind] = rs.decoder(q)
    end
    if plot_red_sol
        plot_object = Plots.surface(1:1:rs.n, 1:1:num_time_steps,transpose(sol_matrix_red[1:rs.n,:]), xlabel="ξ", ylabel="t", title="Reduced solution", titlefontsize=12)
        savefig(plot_object, "plots_temp/reduced_sol_n=$(rs.n).eps")
    end
    if reconst_sol
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, sol_matrix_reconst
    else
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, nothing
    end
end

# CHANGE fn: added method:
# - to compute_reduction_error_analytic_normalized to work with ANALYTIC NORMALIZED DATA (needed for SG examples)
# - to obtain time record
# - to obtain reconstructed solution
# - to plot reduced solution
function compute_reduction_error_analytic_normalized(rs::ReducedSystem, sol_matrix_full; reconst_sol=false, plot_red_sol=false)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_red, time_needed = perform_integration_reduced_normalized(rs; get_time=true)
    sol_matrix_red = zeros(2*rs.n, n_time_steps)
    sol_matrix_reconst = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_red.q)
        sol_matrix_reconst[:, t_ind] = rs.x_ref + rs.decoder(q)
        sol_matrix_red[:, t_ind] = q
    end
    if plot_red_sol
        plot_object = Plots.surface(1:1:rs.n,1:1:n_time_steps,transpose(sol_matrix_red[1:rs.n,:]), xlabel="ξ", ylabel="t", title="Reduced solution", titlefontsize=12)
        savefig(plot_object, "plots_temp/reduced_sol_normalized_n=$(rs.n)_ν=$(rs.params.ν).eps")
    end
    if reconst_sol
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, sol_matrix_reconst
    else
        norm(sol_matrix_reconst - sol_matrix_full)/norm(sol_matrix_full), time_needed, nothing
    end
end

#endregion


#region Section 4: Projection errors

function compute_projection_error(rs::ReducedSystem, sol_full)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_matrix_full = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    norm(rs.decoder(rs.encoder(sol_matrix_full)) - sol_matrix_full)/norm(sol_matrix_full)
end

# CHANGE fn: added method:
# - to compute_reduction_error_analytic to work with NORMALIZED DATA
function compute_projection_error_normalized(rs::ReducedSystem, sol_full)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_matrix_full = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    x_ref_matrix = hcat(fill(rs.x_ref, size(sol_matrix_full,2))...)
    norm(x_ref_matrix + rs.decoder(rs.encoder(sol_matrix_full - x_ref_matrix)) - sol_matrix_full)/norm(sol_matrix_full)
end

# CHANGE fn: added method:
# - to compute_reduction_error_analytic to work with ANALYTIC DATA (needed for SG examples)
function compute_projection_error_analytic(rs::ReducedSystem, sol_matrix_full)
    norm(rs.decoder(rs.encoder(sol_matrix_full)) - sol_matrix_full)/norm(sol_matrix_full)
end


# CHANGE fn: added method:
# - to compute_reduction_error_analytic to work with ANALYTIC NORMALIZED DATA (needed for SG examples)
function compute_projection_error_analytic_normalized(rs::ReducedSystem, sol_matrix_full)
    x_ref_matrix = hcat(fill(rs.x_ref, size(sol_matrix_full,2))...)
    norm(x_ref_matrix + rs.decoder(rs.encoder(sol_matrix_full - x_ref_matrix)) - sol_matrix_full)/norm(sol_matrix_full)
end

#endregion