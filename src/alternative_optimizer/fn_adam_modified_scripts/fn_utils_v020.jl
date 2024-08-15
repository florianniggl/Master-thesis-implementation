# ---------------------------------------------------------------------------------------------------------------------
# fn_utils_v2.jl MODIFIES and EXTENDS utils.jl from GeometricMachineLearning@v0.2.0
# ---------------------------------------------------------------------------------------------------------------------



function apply_toNT(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(p...) for p in zip(ps...))
end


function apply_toNT(fun, any::Any, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(any, p...) for p in zip(ps...))
end