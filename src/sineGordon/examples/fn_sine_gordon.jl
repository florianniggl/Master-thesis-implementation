# ---------------------------------------------------------------------------------------------------------------------
# fn_sine_gordon.jl is a SELF-CREATED file implementing a struct (and constructors) to work with sine-Gordon examples
# ---------------------------------------------------------------------------------------------------------------------



#region Section 1: struct SineGordon

struct SineGordon <: Any
    T_analytic::DataType

    t0::Union{AbstractFloat,Nothing}
    t1::Union{AbstractFloat,Nothing}
    t_steps::Union{Integer,Nothing}

    a::Union{AbstractFloat,Nothing}
    b::Union{AbstractFloat,Nothing}
    N::Union{Integer,Nothing}

    ν_range::Union{StepRangeLen, StepRange, Nothing}

    u₀::Function
    u₁::Function
    φ::Function
    ψ::Function
    exact_solution::Function
end

#endregion


#region Section 2: constructors

# default constructor
function SineGordon(
    a::Union{AbstractFloat,Nothing},
    b::Union{AbstractFloat,Nothing},
    ν_range::Union{StepRangeLen, StepRange, Nothing},
    u₀::Function,
    u₁::Function,
    φ::Function,
    ψ::Function,
    exact_solution::Function;

    T_analytic=T_analytic_global,

    N=N_global,

    t0=start_time_global,
    t1=end_time_global,
    t_steps=time_steps_global)

    SineGordon(T_analytic,t0,t1,t_steps,a,b,N,ν_range,u₀,u₁,φ,ψ,exact_solution)
end

# example1 constructor
function SineGordon_Example1(
    a::Union{AbstractFloat,Nothing},
    b::Union{AbstractFloat,Nothing},
    t0::Union{AbstractFloat,Nothing},
    t1::Union{AbstractFloat,Nothing},
    ν_range::Union{StepRangeLen, StepRange, Nothing};

    T_analytic=T_analytic_global,
    N=N_global,
    t_steps=time_steps_global,
    u₀=example1_u₀,
    u₁=example1_u₁,
    φ=example1_φ,
    ψ=example1_ψ,
    exact_solution=example1_exact_sol)
    
    if a !== nothing
        @assert b > a
    end
    if t0 !== nothing
        @assert t1 > t0
    end

    SineGordon(T_analytic,t0,t1,t_steps,a,b,N,ν_range,u₀,u₁,φ,ψ,exact_solution)
end

# single_soltion constructor
function SineGordon_SingleSoliton(
    a::Union{AbstractFloat,Nothing},
    b::Union{AbstractFloat,Nothing},
    t0::Union{AbstractFloat,Nothing},
    t1::Union{AbstractFloat,Nothing},
    ν_range::Union{StepRangeLen, StepRange, Nothing};

    T_analytic=T_analytic_global,
    N=N_global,
    t_steps=time_steps_global,
    u₀=single_soliton_u₀,
    u₁=single_soliton_u₁,
    φ=single_soliton_φ,
    ψ=single_soliton_ψ,
    exact_solution=single_soliton_exact_sol)

    if a !== nothing
        @assert b > a
    end
    if t0 !== nothing
        @assert t1 > t0
    end

    SineGordon(T_analytic,t0,t1,t_steps,a,b,N,ν_range,u₀,u₁,φ,ψ,exact_solution)
end

# soliton_antisoliton_breather constructor
function SineGordon_SolitonAntisolitonBreather(
    a::Union{AbstractFloat,Nothing},
    b::Union{AbstractFloat,Nothing},
    t0::Union{AbstractFloat,Nothing},
    t1::Union{AbstractFloat,Nothing},
    ν_range::Union{StepRangeLen, StepRange, Nothing};

    T_analytic=T_analytic_global,
    N=N_global,
    t_steps=time_steps_global,
    u₀=soliton_antisoliton_breather_u₀,
    u₁=soliton_antisoliton_breather_u₁,
    φ=soliton_antisoliton_breather_φ,
    ψ=soliton_antisoliton_breather_ψ,
    exact_solution=soliton_antisoliton_breather_exact_sol)

    if a !== nothing
        @assert b > a
    end
    if t0 !== nothing
        @assert t1 > t0
    end

    SineGordon(T_analytic,t0,t1,t_steps,a,b,N,ν_range,u₀,u₁,φ,ψ,exact_solution)
end

# soliton_soliton_doublets constructor
function SineGordon_SolitonSolitonDoublet(
    a::Union{AbstractFloat,Nothing},
    b::Union{AbstractFloat,Nothing},
    t0::Union{AbstractFloat,Nothing},
    t1::Union{AbstractFloat,Nothing},
    ν_range::Union{StepRangeLen, StepRange, Nothing};

    T_analytic=T_analytic_global,
    N=N_global,
    t_steps=time_steps_global,
    u₀=soliton_soliton_doublets_u₀,
    u₁=soliton_soliton_doublets_u₁,
    φ=soliton_soliton_doublets_φ,
    ψ=soliton_soliton_doublets_ψ,
    exact_solution=soliton_soliton_doublets_exact_sol)

    if a !== nothing
        @assert b > a
    end
    if t0 !== nothing
        @assert t1 > t0
    end

    SineGordon(T_analytic,t0,t1,t_steps,a,b,N,ν_range,u₀,u₁,φ,ψ,exact_solution)
end

#endregion