using JuMP
using Memento
using MathOptInterface


"""
Checks whether a JuMPLinearType is constant (and thus has no model associated)
with it. This can only be true if it is an affine expression with no stored
variables.
"""
function is_constant(x::JuMP.AffExpr)
    # TODO (vtjeng): Determine whether there is a built-in function for this as of JuMP>=0.19
    all(values(x.terms) .== 0)
end

function is_constant(x::JuMP.VariableRef)
    false
end

function get_tightening_algorithm(
    x::JuMPLinearType,
    nta::Union{TighteningAlgorithm,Nothing},
)::TighteningAlgorithm
    if is_constant(x)
        return interval_arithmetic
    elseif !(nta === nothing)
        return nta
    else
        # x is not constant, and thus x must have an associated model
        model = owner_model(x)
        return !haskey(model.ext, :MIPVerifyMulti) ? DEFAULT_TIGHTENING_ALGORITHM :
               model.ext[:MIPVerifyMulti].tightening_algorithm
    end
end

@enum BoundType lower_bound_type = -1 upper_bound_type = 1
#! format: off
bound_f = Dict(
    lower_bound_type => lower_bound,
    upper_bound_type => upper_bound
)
bound_obj = Dict(
    lower_bound_type => MathOptInterface.MIN_SENSE,
    upper_bound_type => MathOptInterface.MAX_SENSE
)
bound_delta_f = Dict(
    lower_bound_type => (b, b_0) -> b - b_0,
    upper_bound_type => (b, b_0) -> b_0 - b
)
bound_operator = Dict(
    lower_bound_type => >=,
    upper_bound_type => <=
)
#! format: on

"""
$(SIGNATURES)

Context manager for running `f` on `model`. If `should_relax_integrality` is true, the 
integrality constraints are relaxed before `f` is run and re-imposed after.
"""
function relax_integrality_context(f, model::Model, should_relax_integrality::Bool)
    if should_relax_integrality
        undo_relax = relax_integrality(model)
    end
    r = f(model)
    if should_relax_integrality
        undo_relax()
    end
    return r
end

"""
$(SIGNATURES)

Optimizes the value of `objective` based on `bound_type`, with `b_0`, computed via interval
arithmetic, as a backup.

- If an optimal solution is reached, we return the objective value. We also verify that the 
  objective found is better than the bound `b_0` provided; if this is not the case, we throw an
  error.
- If we reach the user-defined time limit, we compute the best objective bound found. We compare 
  this to `b_0` and return the better result.
- For all other solve statuses, we warn the user and report `b_0`.
"""
function tight_bound_helper(m::Model, bound_type::BoundType, objective::JuMPLinearType, b_0::Number)
    @objective(m, bound_obj[bound_type], objective)
    optimize!(m)
    status = JuMP.termination_status(m)
    if status == MathOptInterface.OPTIMAL
        b = JuMP.objective_value(m)
        db = bound_delta_f[bound_type](b, b_0)
        if db < -1e-8
            Memento.warn(MIPVerifyMulti.LOGGER, "Δb = $(db)")
            ###### REMOVED WARNING (RETURN b_0) #######
            Memento.error(
                MIPVerifyMulti.LOGGER,
                "Δb = $(db). Tightening via interval arithmetic should not give a better result than an optimal optimization.",
            )
            # return b_0
            ###############################
        end
        return b
    elseif status == MathOptInterface.TIME_LIMIT
        return b_0
    else
        Memento.warn(
            MIPVerifyMulti.LOGGER,
            "Unexpected solve status $(status); using interval_arithmetic to obtain bound.",
        )
        return b_0
    end
end

"""
Calculates a tight bound of type `bound_type` on the variable `x` using the specified
tightening algorithm `nta`.

If an upper bound is proven to be below cutoff, or a lower bound is proven to above cutoff,
the algorithm returns early with whatever value was found.
"""
function tight_bound(
    x::JuMPLinearType,
    nta::Union{TighteningAlgorithm,Nothing},
    bound_type::BoundType,
    cutoff::Real,
)
    tightening_algorithm = get_tightening_algorithm(x, nta)
    b_0 = bound_f[bound_type](x)
    if tightening_algorithm == interval_arithmetic ||
       is_constant(x) ||
       bound_operator[bound_type](b_0, cutoff)
        return b_0
    end
    should_relax_integrality = (tightening_algorithm == lp)
    # x is not constant, and thus x must have an associated model
    bound_value = return relax_integrality_context(owner_model(x), should_relax_integrality) do m
        tight_bound_helper(m, bound_type, x, b_0)
    end

    return bound_value
end

function tight_upperbound(
    x::JuMPLinearType;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff::Real = -Inf,
)
    tight_bound(x, nta, upper_bound_type, cutoff)
end

function tight_lowerbound(
    x::JuMPLinearType;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff::Real = Inf,
)
    tight_bound(x, nta, lower_bound_type, cutoff)
end

function log_gap(m::JuMP.Model)
    gap = abs(1 - JuMP.objective_bound(m) / JuMP.objective_value(m))
    Memento.info(
        MIPVerifyMulti.LOGGER,
        "Hit user limit during solve to determine bounds. Multiplicative gap was $gap.",
    )
end

function relu(x::T)::T where {T<:Real}
    return max(zero(T), x)
end

function relu(x::AbstractArray{T}) where {T<:Real}
    return relu.(x)
end

function relu(x::T, l::Real, u::Real)::JuMP.AffExpr where {T<:JuMPLinearType}
    if u < l
        # TODO (vtjeng): This check is in place in case of numerical error in the calculation of bounds.
        # See sample number 4872 (1-indexed) when verified on the lp0.4 network.
        Memento.warn(
            MIPVerifyMulti.LOGGER,
            "Inconsistent upper and lower bounds: u-l = $(u - l) is negative. Attempting to use interval arithmetic bounds instead ...",
        )
        u = upper_bound(x)
        l = lower_bound(x)
    end

    if u <= 0
        # rectified value is always 0
        return zero(T)
    elseif u == l
        return one(T) * l
    elseif u < l
        error(
            MIPVerifyMulti.LOGGER,
            "Inconsistent upper and lower bounds even after using only interval arithmetic: u-l = $(u - l) is negative",
        )
    elseif l >= 0
        # rectified value is always x
        return x
    else
        # since we know that u!=l, x is not constant, and thus x must have an associated model
        model = owner_model(x)
        x_rect = @variable(model)
        a = @variable(model, binary = true)

        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.
        @constraint(model, x_rect <= x + (-l) * (1 - a))
        @constraint(model, x_rect >= x)
        @constraint(model, x_rect <= u * a)
        @constraint(model, x_rect >= 0)

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        set_lower_bound(x_rect, 0)
        set_upper_bound(x_rect, u)
        return x_rect
    end
end

@enum ReLUType split = 0 zero_output = -1 linear_in_input = 1 constant_output = 2

function get_relu_type(l::Real, u::Real)::ReLUType
    if u <= 0
        return zero_output
    elseif u == l
        return constant_output
    elseif l >= 0
        return linear_in_input
    else
        return split
    end
end

struct ReLUInfo
    lowerbounds::Array{Real}
    upperbounds::Array{Real}
end

function Base.show(io::IO, s::ReLUInfo)
    relutypes = get_relu_type.(s.lowerbounds, s.upperbounds)
    print(io, "  Behavior of ReLUs - ")
    for t in instances(ReLUType)
        n = count(x -> x == t, relutypes)
        print(io, "$t: $n")
        if t != last(instances(ReLUType))
            print(io, ", ")
        end
    end
end

"""
Calculates the lower_bound only if `u` is positive; otherwise, returns `u` (since we expect)
the ReLU to be fixed to zero anyway.
"""
function lazy_tight_lowerbound(
    x::JuMPLinearType,
    u::Real;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff = 0,
)::Real
    (u <= cutoff) ? u : tight_lowerbound(x; nta = nta, cutoff = cutoff)
end

function relu(x::JuMPLinearType)::JuMP.AffExpr
    u = tight_upperbound(x, cutoff = 0)
    l = lazy_tight_lowerbound(x, u, cutoff = 0)
    relu(x, l, u)
end

"""
$(SIGNATURES)
Expresses a rectified-linearity constraint: output is constrained to be equal to
`max(x, 0)`.
"""
function relu(
    x::AbstractArray{T};
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
)::Array{JuMP.AffExpr} where {T<:JuMPLinearType}
    show_progress_bar::Bool =
        MIPVerifyMulti.LOGGER.levels[MIPVerifyMulti.LOGGER.level] > MIPVerifyMulti.LOGGER.levels["debug"]
    if !show_progress_bar
        u = tight_upperbound.(x, nta = nta, cutoff = 0)
        l = lazy_tight_lowerbound.(x, u, nta = nta, cutoff = 0)
        return relu.(x, l, u)
    else
        p1 = Progress(length(x), desc = "  Calculating upper bounds: ")
        u = map(x_i -> (next!(p1); tight_upperbound(x_i, nta = nta, cutoff = 0)), x)
        p2 = Progress(length(x), desc = "  Calculating lower bounds: ")
        l = map(v -> (next!(p2); lazy_tight_lowerbound(v..., nta = nta, cutoff = 0)), zip(x, u))

        reluinfo = ReLUInfo(l, u)
        Memento.info(MIPVerifyMulti.LOGGER, "$reluinfo")

        p3 = Progress(length(x), desc = "  Imposing relu constraint: ")
        return x_r = map(v -> (next!(p3); relu(v...)), zip(x, l, u))
    end
end

function masked_relu(x::T, m::Real)::T where {T<:Real}
    if m < 0
        zero(T)
    elseif m > 0
        x
    else
        relu(x)
    end
end

function masked_relu(x::AbstractArray{<:Real}, m::AbstractArray{<:Real})
    masked_relu.(x, m)
end

function masked_relu(x::T, m::Real)::JuMP.AffExpr where {T<:JuMPLinearType}
    if m < 0
        zero(T)
    elseif m > 0
        x
    else
        relu(x)
    end
end

"""
$(SIGNATURES)
Expresses a masked rectified-linearity constraint, with three possibilities depending on
the value of the mask. Output is constrained to be:
```
1) max(x, 0) if m=0,
2) 0 if m<0
3) x if m>0
```
"""
function masked_relu(
    x::AbstractArray{<:JuMPLinearType},
    m::AbstractArray{<:Real};
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
)::Array{JuMP.AffExpr}
    @assert(size(x) == size(m))
    s = size(m)
    # We add the constraints corresponding to the active ReLUs to the model
    zero_idx = Iterators.filter(i -> m[i] == 0, CartesianIndices(s)) |> collect
    d = Dict(zip(zero_idx, relu(x[zero_idx], nta = nta)))

    # We determine the output of the masked relu, which is either:
    #  1) the output of the relu that we have previously determined when adding the
    #     constraints to the model.
    #  2, 3) the result of applying the (elementwise) masked_relu function.
    return map(i -> m[i] == 0 ? d[i] : masked_relu(x[i], m[i]), CartesianIndices(s))
end

function maximum(xs::AbstractArray{T}, d)::T where {T<:Real}
    max = Base.maximum(xs)
    return (max, Bool[x == max for x in 1:length(xs)], 1:length(xs), [], [], d)
end

function minimum(xs::AbstractArray{T}, d)::T where {T<:Real}
    min = Base.minimum(xs)
    return (min, Bool[x == min for x in 1:length(xs)], 1:length(xs), [], [], d)
end

function maximum_of_constants(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert all(is_constant.(xs))
    max_val = map(x -> x.constant, xs) |> maximum
    return one(JuMP.VariableRef) * max_val
end

function minimum_of_constants(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert all(is_constant.(xs))
    min_val = map(x -> x.constant, xs) |> minimum
    return one(JuMP.VariableRef) * min_val
end

function second_maximum_of_constants(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert all(is_constant.(xs))
    consts_array = map(x -> x.constant, xs)
    sorted_const_array = consts_array |> sort
    return one(JuMP.VariableRef) * sorted_const_array[length(sorted_const_array)-1]
end

function kth_maximum_of_constants(xs::AbstractArray{T}, k::Integer) where {T<:JuMPLinearType}
    @assert all(is_constant.(xs))
    consts_array = map(x -> x.constant, xs)
    sorted_const_array = consts_array |> sort
    return one(JuMP.VariableRef) * sorted_const_array[length(sorted_const_array)-k+1]
end

"""
$(SIGNATURES)
Expresses a maximization constraint: output is constrained to be equal to `max(xs)`.
"""
function maximum(xs::AbstractArray{T}, labels, d) where {T<:JuMPLinearType}

    @assert length(xs) > 0

    if length(xs) == 1
        return (xs[1], [1], 1:length(xs), [], [], d)
    end

    if all(is_constant.(xs))
        max = maximum_of_constants(xs)
        return (max, Bool[x == max for x in map(x_ -> x_.constant, xs)], 1:length(xs), [], [], d)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)

    set_time_limit_sec(model, 20)

    print("\n\nCalculating xs bounds for max var\n")
    # TODO (vtjeng): [PERF] skip calculating lower_bound for index if upper_bound is lower than
    # largest current lower_bound.
    p1 = Progress(length(xs), desc = "  Calculating upper bounds: ")
    us = map(x_i -> (next!(p1); d[:Bounds][x_i][:"ub"]==Inf ? tight_upperbound(x_i) : d[:Bounds][x_i][:"ub"]), xs)
    p2 = Progress(length(xs), desc = "  Calculating lower bounds: ")
    ls = map(x_i -> (next!(p2); d[:Bounds][x_i][:"lb"]==-Inf ? tight_lowerbound(x_i) : d[:Bounds][x_i][:"lb"]), xs)
    print("\nDone Calculating xs bounds for max var\n\n")

    for (i, x_i) in enumerate(xs)
        d[:Bounds][x_i][:"ub"] = us[i]
        d[:Bounds][x_i][:"lb"] = ls[i]
    end

    GRBreset(Ref(model), 1)

    l = Base.maximum(ls)
    u = Base.maximum(us)

    # print("\nlower-bounds: $ls\n")
    # print("upper-bounds: $us\n")

    if l == u
        return (one(T) * l, Bool[x == l for x in ls], 1:length(xs), [], [], d)
        Memento.info(MIPVerifyMulti.LOGGER, "Output of maximum is constant.")
    end
    # at least one index will satisfy this property because of check above.
    filtered_indexes = us .> l

    # TODO (vtjeng): Smarter log output if maximum function is being used more than once (for example, in a max-pooling layer).
    Memento.info(
        MIPVerifyMulti.LOGGER,
        "Number of inputs to maximum function possibly taking maximum value: $(filtered_indexes |> sum)",
    )

    (maximum_nontarget_var, maximum_nontarget_arr, vars, consts) = 
    maximum(xs[filtered_indexes], ls[filtered_indexes], us[filtered_indexes])

    return (maximum_nontarget_var, maximum_nontarget_arr, filtered_indexes, vars, consts, d)

end

function minimum(xs::AbstractArray{T}, labels, d) where {T<:JuMPLinearType}
    if length(xs) == 1
        return (xs[1], [1], 1:length(xs), [], [], d)
    end

    if all(is_constant.(xs))
        min = minimum_of_constants(xs)
        return (min, Bool[x == min for x in map(x_ -> x_.constant, xs)], 1:length(xs), [], [], d)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)

    set_time_limit_sec(model, 20)

    print("\n\nCalculating xs bounds for min var\n")
    # TODO (vtjeng): [PERF] skip calculating lower_bound for index if upper_bound is lower than
    # largest current lower_bound.
    p1 = Progress(length(xs), desc = "  Calculating upper bounds: ")
    us = map(x_i -> (next!(p1); d[:Bounds][x_i][:"ub"]==Inf ? tight_upperbound(x_i) : d[:Bounds][x_i][:"ub"]), xs)
    p2 = Progress(length(xs), desc = "  Calculating lower bounds: ")
    ls = map(x_i -> (next!(p2); d[:Bounds][x_i][:"lb"]==-Inf ? tight_lowerbound(x_i) : d[:Bounds][x_i][:"lb"]), xs)
    print("\nDone Calculating xs bounds for min var\n\n")

    for (i, x_i) in enumerate(xs)
        d[:Bounds][x_i][:"ub"] = us[i]
        d[:Bounds][x_i][:"lb"] = ls[i]
    end

    GRBreset(Ref(model), 1)

    l = Base.minimum(ls)
    u = Base.minimum(us)

    # print("\nlower-bounds: $ls\n")
    # print("upper-bounds: $us\n")

    if l == u
        return (one(T) * l, Bool[x == l for x in ls], 1:length(xs), [], [], d)
        Memento.info(MIPVerifyMulti.LOGGER, "Output of minimum is constant.")
    end
    # at least one index will satisfy this property because of check above.
    filtered_indexes = ls .< u

    # TODO (vtjeng): Smarter log output if maximum function is being used more than once (for example, in a max-pooling layer).
    Memento.info(
        MIPVerifyMulti.LOGGER,
        "Number of inputs to minimum function possibly taking minimum value: $(filtered_indexes |> sum)",
    )

    (minimum_nontarget_var, minimum_nontarget_arr, vars, consts) = 
    minimum(xs[filtered_indexes], ls[filtered_indexes], us[filtered_indexes])

    return (minimum_nontarget_var, minimum_nontarget_arr, filtered_indexes, vars, consts, d)

end

"""
$(SIGNATURES)
Expresses a second-maximization constraint.
"""
function second_maximum(xs::AbstractArray{T})::JuMP.AffExpr where {T<:JuMPLinearType}
    if length(xs) == 1
        return xs[1]
    end

    if all(is_constant.(xs))
        return second_maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)

    # TODO (vtjeng): [PERF] skip calculating lower_bound for index if upper_bound is lower than
    # largest current lower_bound.
    p1 = Progress(length(xs), desc = "  Calculating upper bounds: ")
    us = map(x_i -> (next!(p1); tight_upperbound(x_i)), xs)
    p2 = Progress(length(xs), desc = "  Calculating lower bounds: ")
    ls = map(x_i -> (next!(p2); tight_lowerbound(x_i)), xs)

    l = Base.maximum(ls)
    u = Base.maximum(us)

    if l == u
        return one(T) * l
        Memento.info(MIPVerifyMulti.LOGGER, "Output of second-maximum is constant.")
    end
    # at least one index will satisfy this property because of check above.
    filtered_indexes = us .> l

    # TODO (vtjeng): Smarter log output if maximum function is being used more than once (for example, in a max-pooling layer).
    Memento.info(
        MIPVerifyMulti.LOGGER,
        "Number of inputs to second-maximum function possibly taking second-maximum value: $(filtered_indexes |> sum)",
    )

    return second_maximum(xs[filtered_indexes], ls[filtered_indexes], us[filtered_indexes])
end

function kth_maximum(xs::AbstractArray{T}, labels, k::Integer, d) where {T<:JuMPLinearType}
    if length(xs) == 1
        return (xs[1], [], [], d)
    end

    if all(is_constant.(xs))
        return (kth_maximum_of_constants(xs, k), [], [], d)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)

    set_time_limit_sec(model, 20)

    print("\n\nCalculating xs bounds for kth-max var\n")
    # TODO (vtjeng): [PERF] skip calculating lower_bound for index if upper_bound is lower than
    # largest current lower_bound.
    p1 = Progress(length(xs), desc = "  Calculating upper bounds: ")
    us = map(x_i -> (next!(p1); d[:Bounds][x_i][:"ub"]==Inf ? tight_upperbound(x_i) : d[:Bounds][x_i][:"ub"]), xs)
    p2 = Progress(length(xs), desc = "  Calculating lower bounds: ")
    ls = map(x_i -> (next!(p2); d[:Bounds][x_i][:"lb"]==-Inf ? tight_lowerbound(x_i) : d[:Bounds][x_i][:"lb"]), xs)
    print("\nDone Calculating xs bounds for kth-max var\n\n")

    for (i, x_i) in enumerate(xs)
        d[:Bounds][x_i][:"ub"] = us[i]
        d[:Bounds][x_i][:"lb"] = ls[i]
    end

    GRBreset(Ref(model), 1)

    l = kth_maximum_array(ls, k)
    u = kth_maximum_array(us, k)

    # print("\nlower-bounds: $ls\n")
    # print("upper-bounds: $us\n")

    if l == u
        return (one(T) * l, [], [], d)
        Memento.info(MIPVerifyMulti.LOGGER, "Output of kth-maximum is constant.")
    end
    # at least one index will satisfy this property because of check above.
    filtered_indexes = us .> l

    # TODO (vtjeng): Smarter log output if maximum function is being used more than once (for example, in a max-pooling layer).
    Memento.info(
        MIPVerifyMulti.LOGGER,
        "Number of inputs to kth-maximum function possibly taking 1-kth-maximum values: $(filtered_indexes |> sum)",
    )

    (kth_x_max, vars, consts) = kth_maximum(xs[filtered_indexes], ls[filtered_indexes], us[filtered_indexes], k)

    return (kth_x_max, vars, consts, d)
end

function maximum(
    xs::AbstractArray{T,1},
    ls::AbstractArray{<:Real,1},
    us::AbstractArray{<:Real,1},
) where {T<:JuMPLinearType}

    @assert length(xs) > 0
    @assert length(xs) == length(ls)
    @assert length(xs) == length(us)

    vars = []
    consts = []

    if all(is_constant.(xs))
        max = maximum_of_constants(xs)
        return (max, Bool[x == max for x in map(x_ -> x_.constant, xs)],[], [])
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    if length(xs) == 1
        return (first(xs), [1], vars, consts)
    else
        l = Base.maximum(ls)
        u = Base.maximum(us)
        x_max = @variable(model, lower_bound = l, upper_bound = u)
        push!(vars, x_max)
        a = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(a) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, a[i])
            umaxi = Base.maximum(us[1:end.!=i])
            push!(consts, @constraint(model, x_max <= x + (1 - a[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, x_max >= x))
        end
        return (x_max, a, vars, consts)
    end
end

function minimum(
    xs::AbstractArray{T,1},
    ls::AbstractArray{<:Real,1},
    us::AbstractArray{<:Real,1},
) where {T<:JuMPLinearType}

    @assert length(xs) > 0
    @assert length(xs) == length(ls)
    @assert length(xs) == length(us)

    vars = []
    consts = []

    if all(is_constant.(xs))
        min = minimum_of_constants(xs)
        return (min, Bool[x == min for x in map(x_ -> x_.constant, xs)],[], [])
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    if length(xs) == 1
        return (first(xs), [1], vars, consts)
    else
        l = Base.minimum(ls)
        u = Base.minimum(us)
        x_min = @variable(model, lower_bound = l, upper_bound = u)
        push!(vars, x_min)
        a = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(a) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, a[i])
            lmini = Base.minimum(ls[1:end.!=i])
            push!(consts, @constraint(model, x_min >= x + (1 - a[i]) * (lmini - us[i])))
            push!(consts, @constraint(model, x_min <= x))
        end
        return (x_min, a, vars, consts)
    end
end

function second_maximum_array(arr::AbstractArray)
    sorted = arr |> sort
    if length(sorted)==1
        return sorted[1]
    else
        return sorted[length(arr)-1]
    end
end

function kth_maximum_array(arr::AbstractArray, k::Integer)
    sorted = arr |> sort
    if length(sorted)==1
        return sorted[1]
    else
        return sorted[length(arr)-k+1]
    end
end


function second_maximum(
    xs::AbstractArray{T,1},
    ls::AbstractArray{<:Real,1},
    us::AbstractArray{<:Real,1},
)::JuMP.AffExpr where {T<:JuMPLinearType}

    @assert length(xs) > 0
    @assert length(xs) == length(ls)
    @assert length(xs) == length(us)

    if all(is_constant.(xs))
        return second_maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    if length(xs) == 1
        return first(xs)
    else
        l = Base.maximum(ls)
        u = Base.maximum(us)
        x_max = @variable(model, lower_bound = l, upper_bound = u)
        a = @variable(model, [1:length(xs)], binary = true)
        @constraint(model, sum(a) == 1)
        for (i, x) in enumerate(xs)
            umaxi = Base.maximum(us[1:end.!=i])
            @constraint(model, x_max <= x + (1 - a[i]) * (umaxi - ls[i]))
            @constraint(model, x_max >= x)
        end
        
        l = second_maximum_array(ls)
        u = second_maximum_array(us)
        second_x_max = @variable(model, lower_bound = l, upper_bound = u)
        b = @variable(model, [1:length(xs)], binary = true)
        @constraint(model, sum(b) == 1)
        l_min = Base.minimum(ls)
        for (i, x) in enumerate(xs)
            umaxi = second_maximum_array(us[1:end.!=i])
            @constraint(model, second_x_max <= x + (1 - b[i]) * (umaxi - ls[i]))
            @constraint(model, second_x_max >= x * (1-a[i]) + l_min * a[i])
            @constraint(model, a[i] + b[i] <= 1)
        end
        return second_x_max
    end
end

function kth_maximum(
    xs::AbstractArray{T,1},
    ls::AbstractArray{<:Real,1},
    us::AbstractArray{<:Real,1},
    k::Integer,
) where {T<:JuMPLinearType}

    vars = []
    consts = []

    @assert length(xs) > 0
    @assert length(xs) == length(ls)
    @assert length(xs) == length(us)

    @assert (k==1 || k==2 || k==3 || k==4 || k==5 || k==6 || k==7 || k==8)

    if all(is_constant.(xs))
        return (kth_maximum_of_constants(xs, k), vars, consts)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)

    if length(xs) == 1
        return (first(xs), vars, consts)
    end

    l = Base.maximum(ls)
    u = Base.maximum(us)
    x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, x_max)
    a = @variable(model, [1:length(xs)], binary = true)
    push!(consts, @constraint(model, sum(a) == 1))
    for (i, x) in enumerate(xs)
        push!(vars, a[i])
        umaxi = Base.maximum(us[1:end.!=i])
        push!(consts, @constraint(model, x_max <= x + (1 - a[i]) * (umaxi - ls[i])))
        push!(consts, @constraint(model, x_max >= x))
    end

    if k==1
        return (x_max, vars, consts)
    end

    l_min = Base.minimum(ls)
    
    l = kth_maximum_array(ls, 2)
    u = kth_maximum_array(us, 2)
    second_x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, second_x_max)
    if length(xs) == 2
        for (i, x) in enumerate(xs)
            # @constraint(model, (a[i] == 0) => {second_x_max == x})
            push!(consts, @constraint(model, second_x_max <= x))
            push!(consts, @constraint(model, second_x_max >= x * (1-a[i]) + l_min * a[i]))
        end
    else
        b = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(b) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, b[i])
            umaxi = Base.maximum(us[1:end.!=i])
            push!(consts, @constraint(model, second_x_max <= x + (1 - b[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, second_x_max >= x * (1-a[i]) + l_min * a[i]))
            push!(consts, @constraint(model, a[i] + b[i] <= 1))
        end
    end

    if k==2
        return (second_x_max, vars, consts)
    end

    l = kth_maximum_array(ls, 3)
    u = kth_maximum_array(us, 3)
    third_x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, third_x_max)
    if length(xs) == 3
        for (i, x) in enumerate(xs)
            # @constraint(model, (a[i] + b[i] == 0) => {third_x_max == x})
            push!(consts, @constraint(model, third_x_max <= x))
            push!(consts, @constraint(model, third_x_max >= x * (1-a[i]-b[i]) + l_min * (a[i]+b[i])))
        end
    else
        c = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(c) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, c[i])
            umaxi = Base.maximum(us[1:end.!=i])
            push!(consts, @constraint(model, third_x_max <= x + (1 - c[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, third_x_max >= x * (1-a[i]-b[i]) + l_min * (a[i]+b[i])))
            push!(consts, @constraint(model, a[i] + b[i] + c[i] <= 1))
        end
    end

    if k==3
        return (third_x_max, vars, consts)
    end

    l = kth_maximum_array(ls, 4)
    u = kth_maximum_array(us, 4)
    fourth_x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, fourth_x_max)
    if length(xs) == 4
        for (i, x) in enumerate(xs)
            # @constraint(model, (a[i] + b[i] + c[i] == 0) => {fourth_x_max == x})
            push!(consts, @constraint(model, fourth_x_max <= x))
            push!(consts, @constraint(model, fourth_x_max >= x * (1-a[i]-b[i]-c[i]) + l_min * (a[i]+b[i]+c[i])))
        end
    else
        d = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(d) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, d[i])
            umaxi = Base.maximum(us[1:end.!=i])
            push!(consts, @constraint(model, fourth_x_max <= x + (1 - d[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, fourth_x_max >= x * (1-a[i]-b[i]-c[i]) + l_min * (a[i]+b[i]+c[i])))
            push!(consts, @constraint(model, a[i] + b[i] + c[i] + d[i] <= 1))
        end
    end

    if k==4
        return (fourth_x_max, vars, consts)
    end

    l = kth_maximum_array(ls, 5)
    u = kth_maximum_array(us, 5)
    fifth_x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, fifth_x_max)
    if length(xs) == 5
        for (i, x) in enumerate(xs)
            # @constraint(model, (a[i] + b[i] + c[i] == 0) => {fourth_x_max == x})
            push!(consts, @constraint(model, fifth_x_max <= x))
            push!(consts, @constraint(model, fifth_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]) + l_min * (a[i]+b[i]+c[i]+d[i])))
        end
    else
        e = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(e) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, e[i])
            umaxi = Base.maximum(us[1:end.!=i])
            push!(consts, @constraint(model, fifth_x_max <= x + (1 - e[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, fifth_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]) + l_min * (a[i]+b[i]+c[i]+d[i])))
            push!(consts, @constraint(model, a[i] + b[i] + c[i] + d[i] + e[i] <= 1))
        end
    end

    if k==5
        return (fifth_x_max, vars, consts)
    end

    l = kth_maximum_array(ls, 6)
    u = kth_maximum_array(us, 6)
    sixth_x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, sixth_x_max)
    if length(xs) == 6
        for (i, x) in enumerate(xs)
            # @constraint(model, (a[i] + b[i] + c[i] == 0) => {fourth_x_max == x})
            push!(consts, @constraint(model, sixth_x_max <= x))
            push!(consts, @constraint(model, sixth_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]-e[i]) + l_min * (a[i]+b[i]+c[i]+d[i]+e[i])))
        end
    else
        f = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(f) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, f[i])
            push!(consts, @constraint(model, sixth_x_max <= x + (1 - f[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, sixth_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]-e[i]) + l_min * (a[i]+b[i]+c[i]+d[i]+e[i])))
            push!(consts, @constraint(model, a[i] + b[i] + c[i] + d[i] + e[i] + f[i] <= 1))
        end
    end

    if k==6
        return (sixth_x_max, vars, consts)
    end

    l = kth_maximum_array(ls, 7)
    u = kth_maximum_array(us, 7)
    seventh_x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, seventh_x_max)
    if length(xs) == 7
        for (i, x) in enumerate(xs)
            # @constraint(model, (a[i] + b[i] + c[i] == 0) => {fourth_x_max == x})
            push!(consts, @constraint(model, seventh_x_max <= x))
            push!(consts, @constraint(model, seventh_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]-e[i]-f[i]) + l_min * (a[i]+b[i]+c[i]+d[i]+e[i]+f[i])))
        end
    else
        g = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(g) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, g[i])
            umaxi = Base.maximum(us[1:end.!=i])
            push!(consts, @constraint(model, seventh_x_max <= x + (1 - g[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, seventh_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]-e[i]-f[i]) + l_min * (a[i]+b[i]+c[i]+d[i]+e[i]+f[i])))
            push!(consts, @constraint(model, a[i] + b[i] + c[i] + d[i] + e[i] + f[i] + g[i] <= 1))
        end
    end

    if k==7
        return (seventh_x_max, vars, consts)
    end

    l = kth_maximum_array(ls, 8)
    u = kth_maximum_array(us, 8)
    eighth_x_max = @variable(model, lower_bound = l, upper_bound = u)
    push!(vars, eighth_x_max)
    if length(xs) == 8
        for (i, x) in enumerate(xs)
            # @constraint(model, (a[i] + b[i] + c[i] == 0) => {fourth_x_max == x})
            push!(consts, @constraint(model, eighth_x_max <= x))
            push!(consts, @constraint(model, eighth_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]-e[i]-f[i]-g[i]) + l_min * (a[i]+b[i]+c[i]+d[i]+e[i]+f[i]+g[i])))
        end
    else
        h = @variable(model, [1:length(xs)], binary = true)
        push!(consts, @constraint(model, sum(h) == 1))
        for (i, x) in enumerate(xs)
            push!(vars, h[i])
            umaxi = Base.maximum(us[1:end.!=i])
            push!(consts, @constraint(model, eighth_x_max <= x + (1 - h[i]) * (umaxi - ls[i])))
            push!(consts, @constraint(model, eighth_x_max >= x * (1-a[i]-b[i]-c[i]-d[i]-e[i]-f[i]-g[i]) + l_min * (a[i]+b[i]+c[i]+d[i]+e[i]+f[i]+g[i])))
            push!(consts, @constraint(model, a[i] + b[i] + c[i] + d[i] + e[i] + f[i] + g[i] + h[i] <= 1))
        end
    end

    if k==8
        return (eighth_x_max, vars, consts)
    end

end

"""
$(SIGNATURES)
Expresses a one-sided maximization constraint: output is constrained to be at least
`max(xs)`.

Only use when you are minimizing over the output in the objective.

NB: If all of xs are constant, we simply return the largest of them.
"""
function maximum_ge(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert length(xs) > 0
    if all(is_constant.(xs))
        return (maximum_of_constants(xs), [], [])
    end
    if length(xs) == 1
        return (first(xs), [], [])
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    x_max = @variable(model)
    consts = [@constraint(model, x_max .>= xs)]
    return (x_max, [x_max], consts)
end

function minimum_ge(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert length(xs) > 0
    if all(is_constant.(xs))
        return (minimum_of_constants(xs), [], [])
    end
    if length(xs) == 1
        return (first(xs), [], [])
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    x_min = @variable(model)
    consts = [@constraint(model, x_min .<= xs)]
    return (x_min, [x_min], consts)
end

"""
$(SIGNATURES)
Expresses a one-sided absolute-value constraint: output is constrained to be at least as
large as `|x|`.

Only use when you are minimizing over the output in the objective.
"""
function abs_ge(x::JuMPLinearType)::JuMP.AffExpr
    if is_constant(x)
        return one(JuMP.VariableRef) * abs(x.constant)
    end
    model = owner_model(x)
    u = upper_bound(x)
    l = lower_bound(x)
    if u <= 0
        return -x
    elseif l >= 0
        return x
    else
        x_abs = @variable(model)
        @constraint(model, x_abs >= x)
        @constraint(model, x_abs >= -x)
        set_lower_bound(x_abs, 0)
        set_upper_bound(x_abs, max(-l, u))
        return x_abs
    end
end

function get_target_indexes(
    target_index::Integer,
    array_length::Integer;
    invert_target_selection::Bool = false,
)

    get_target_indexes(
        [target_index],
        array_length,
        invert_target_selection = invert_target_selection,
    )

end

function get_target_indexes(
    target_indexes::Array{<:Integer,1},
    array_length::Integer;
    invert_target_selection::Bool = false,
    )::AbstractArray{<:Integer,1}

    @assert length(target_indexes) >= 1
    @assert all(target_indexes .>= 1) && all(target_indexes .<= array_length)

    invert_target_selection ? filter((x) -> x ∉ target_indexes, 1:array_length) : target_indexes
end

function get_vars_for_max_index(
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1},
    )::Tuple{JuMPLinearType,Array{<:JuMPLinearType,1}}

    @assert length(xs) >= 1

    target_vars = xs[Bool[i ∈ target_indexes for i in 1:length(xs)]]
    nontarget_vars = xs[Bool[i ∉ target_indexes for i in 1:length(xs)]]

    maximum_target_var = length(target_vars) == 1 ? target_vars[1] : MIPVerifyMulti.maximum(target_vars)

    return (maximum_target_var, nontarget_vars)
end

function get_vars_for_second_max_index(
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1},
    )::Tuple{JuMPLinearType,Array{<:JuMPLinearType,1}}

    @assert length(xs) >= 1

    target_vars = xs[Bool[i ∈ target_indexes for i in 1:length(xs)]]
    nontarget_vars = xs[Bool[i ∉ target_indexes for i in 1:length(xs)]]
    second_maximum_target_var = length(target_vars) == 1 ? target_vars[1] : MIPVerifyMulti.second_maximum(target_vars)

    return (second_maximum_target_var, nontarget_vars)
end

function get_vars_for_kth_max_index(
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1},
    k::Integer,
    d,
    )

    @assert length(xs) >= 1

    target_vars = xs[Bool[i ∈ target_indexes for i in 1:length(xs)]]
    nontarget_vars = xs[Bool[i ∉ target_indexes for i in 1:length(xs)]]

    @assert length(target_vars) >= k

    (kth_maximum_target_var, vars, consts, d) = MIPVerifyMulti.kth_maximum(target_vars, target_indexes, k, d)

    return (kth_maximum_target_var, nontarget_vars, vars, consts, d)
end

function add_constraints_buffer(const_buf, xs::Array{<:JuMPLinearType,1},)
    model = owner_model(xs)
    for constraint in const_buf
        l1 = constraint["l1"]
        l2 = constraint["l2"]
        if constraint["relation"] == ">" || constraint["relation"] == ">="
            @constraint(model, xs[l1] >= xs[l2])
        elseif constraint["relation"] == "<" || constraint["relation"] == "<="
            @constraint(model, xs[l1] <= xs[l2])
        elseif constraint["relation"] == "=="
            @constraint(model, xs[l1] == xs[l2])
        else
            error("wrong relation type in constraint.")
        end
    end
end

function get_vars_for_non_max_indexes(
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1},
    )::Tuple{JuMPLinearType,Array{<:JuMPLinearType,1}}

    @assert length(xs) >= 1

    target_vars = xs[Bool[i ∈ target_indexes for i in 1:length(xs)]]
    nontarget_vars = xs[Bool[i ∉ target_indexes for i in 1:length(xs)]]

    maximum_target_var = length(target_vars) == 1 ? target_vars[1] : MIPVerifyMulti.maximum(target_vars)
    non_max_target_vars = target_vars[Bool[t_v != maximum_target_var for t_v in target_vars]]

    return (non_max_target_vars, nontarget_vars)
end


"""
$(SIGNATURES)

Imposes constraints ensuring that one of the elements at the target_indexes is the
largest element of the array x. More specifically, we require `x[j] - x[i] ≥ margin` for
some `j ∈ target_indexes` and for all `i ∉ target_indexes`.
In the multilabel case, it imposes constraints ensuring that the winning indexes are 
a subset of the target_indexes (the winning are the n indexes (or less) that have the maximum values and are above threshold).
"""
function set_max_indexes(
    model::Model,
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1};
    multilabel::Integer = 1,
    margin::Real = 0,
    threshold::Real=0.1,    # this was for the above threshold constraint (not needed anymore, now it takes k maxes)
    )::Nothing

    (maximum_target_var, nontarget_vars) = get_vars_for_max_index(xs, target_indexes)
    (kth_maximum_target_var, nontarget_vars) = get_vars_for_kth_max_index(xs, target_indexes, multilabel)

    if multilabel == 1
        @constraint(model, nontarget_vars .<= maximum_target_var - margin)
    else
        # @variable(model, nontargets_under_thresh, binary = true)
        # @variable(model, nontarget_above_thresh, binary = true)
        # @constraint(model, nontargets_under_thresh => {nontarget_vars <= threshold})
        # @constraint(model, nontarget_vars .<= maximum_target_var - margin)
        @constraint(model, nontarget_vars .<= kth_maximum_target_var - margin)
        # @constraint(model, nontargets_under_thresh + nontarget_above_thresh == 1)
    end
    
    return nothing
end

function get_first_true_index(xs)
    
    for (i, a_i) in enumerate(xs)
        if value.(a_i) == 1
            return i
        end
    end

end

function get_first_true_index(
    xs::AbstractArray{T,1},
) where {T<:Real}

    for (i, a_i) in enumerate(xs)
        if a_i == 1
            return i
        end
    end

end
