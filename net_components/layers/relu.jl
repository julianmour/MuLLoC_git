export ReLU

"""
$(TYPEDEF)

Represents a ReLU operation.

`p(x)` is shorthand for [`relu(x)`](@ref) when `p` is an instance of
`ReLU`.
"""
struct ReLU <: Layer
    tightening_algorithm::Union{TighteningAlgorithm,Nothing}
end

ReLU() = ReLU(nothing)

function Base.show(io::IO, p::ReLU)
    print(io, "ReLU()")
end

(p::ReLU)(x::Array{<:Real}) = relu(x)
(p::ReLU)(x::Array{<:JuMPLinearType}) =
    (Memento.info(MIPVerifyMulti.LOGGER, "Applying $p ..."); relu(x, nta = p.tightening_algorithm))
