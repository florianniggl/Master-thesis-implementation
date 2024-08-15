# ---------------------------------------------------------------------------------------------------------------------
# fn_hamiltonian_system.jl is a SELF-CREATED containing all computations for the discretized SG-hamiltonian system
# ---------------------------------------------------------------------------------------------------------------------



function LinOp(params)
    dim = params.N 
    h_N2 = 1/params.h_N^2
    super=fill(1,dim - 1)
    sub=fill(1,dim - 1)
    main=fill(-2,dim)
    h_N2 * Tridiagonal(sub,main,super)
end

function nonlinearity_f(params)
    function f(q, t)
        h_N2 = params.h_N^2
        v = sin.(q)
        v[firstindex(v)] -= params.φ(t) / h_N2
        v[lastindex(v)] -= params.ψ(t) / h_N2
        return v 
    end
    return f
end

function v_field(params)
    L = LinOp(params)
    f = nonlinearity_f(params)
    N=params.N
    # note: params are only passed for complicance with function from reduced_system.jl, can be changed!
    function v(v, t, x, params)
        v .= vcat( x[N+1:2*N], L*x[1:N] - f(x[1:N],t)) 
    end
    v 
end

function v_field_explicit(params)
    L = LinOp(params)
    f = nonlinearity_f(params)
    N=params.N
    # note: params are only passed for complicance with function from reduced_system.jl, can be changed!
    function v(t, x, params)
        vcat( x[N+1:2*N], L*x[1:N] - f(x[1:N],t)) 
    end
    v 
end

