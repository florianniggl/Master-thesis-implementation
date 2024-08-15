# ---------------------------------------------------------------------------------------------------------------------
# fn_customtypes.jl is a SELF-CREATED file defining custom types to enable choosing between different metrics 
# (for calculating the Riemannian gradient) and different vector transports
# ---------------------------------------------------------------------------------------------------------------------



abstract type Metric end

struct Euclidean <: Metric end
struct Canonical <: Metric end


abstract type Vectortransport end

struct SubmanifoldVectortransport <: Vectortransport end
struct DifferentialVectortransport <: Vectortransport end


abstract type Loss end 

struct LossGML <: Loss end
struct LossModified <: Loss end
