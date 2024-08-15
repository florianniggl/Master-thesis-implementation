# ---------------------------------------------------------------------------------------------------------------------
# fn_vector_fields_v2.jl is a MODIFICATION of vector_fields.jl from GeometricMachineLearning@v0.2.0
# Note: the original file is not part of the source code of GeometricMachineLearning.jl release package, instead it is 
# part of test setups in scripts/symplectic_autoencoders
# ---------------------------------------------------------------------------------------------------------------------



function v_f_hamiltonian(params)
    K = assemble_matrix(params.μ, params.Δx, params.Ñ)
    function f(f, t, q, p, params)
        f .= - (K.parent + K.parent') * q / params.Δx
    end
    function v(v, t, q, p, params)
        v .= params.Δx * p / params.Δx
    end
    function hamiltonian(t, q, p, params)
        q'*K.parent*q + eltype(q)(.5) * params.Δx * p'*p
    end
    (v, f, hamiltonian)
end


function v_field(params)
    K = assemble_matrix(params.μ, params.Δx, params.Ñ).parent / params.Δx
    M = -(K+K')
    N = params.Ñ + 2
    function v(v, t, q, params)
        v .= vcat(q[N+1:2*N], M*q[1:N])  
    end
    v 
end

function v_field_explicit(params)
    K = assemble_matrix(params.μ, params.Δx, params.Ñ).parent / params.Δx 
    M = -(K+K')
    N = params.Ñ + 2
    function v(t, q, params)
        vcat(q[N+1:2*N], M*q[1:N])
    end
    v 
end