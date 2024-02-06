module MIPVerifyMulti

using Base.Cartesian
using JuMP
using JLD2
using MathOptInterface
using Memento
using DocStringExtensions
using ProgressMeter

# TODO: more reliable way to determine location for dependencies
const dependencies_path = joinpath(@__DIR__, "..", "deps")

export find_adversarial_example, frac_correct, interval_arithmetic, lp, mip, get_norm, closest, worst

@enum TighteningAlgorithm interval_arithmetic = 1 lp = 2 mip = 3
@enum AdversarialExampleObjective closest = 1 worst = 2
const DEFAULT_TIGHTENING_ALGORITHM = mip
# const DEFAULT_TIGHTENING_ALGORITHM = interval_arithmetic
# const DEFAULT_TIGHTENING_ALGORITHM = lp

# importing vendor/ConditionalJuMP.jl first as the remaining files use functions
# defined in it. we're unsure if this is necessary.
include("vendor/ConditionalJuMP.jl")
include("net_components.jl")
include("models.jl")
include("utils.jl")
include("logging.jl")

function get_max_index(x::Array{<:Real,1})::Integer
    return findmax(x)[2]
end


# this function returns the n indexes (or less) that have the maximum values and are above threshold
# this is used for multi-label classification
function get_multilabel_indexes(x::Array{<:Real,1}, n::Integer, threshold::Real=0.1)::Set{Integer}
    vals = copy(x)
    result = Set{Integer}()
    if n<=0
        return result
    end
    for i in range(1,n)
        (value, index) = findmax(vals)
        if value > threshold
            push!(result, index)
            vals[index] = threshold
        else
            return result
        end
    end
    return result
end

function get_n_maxes_indexes(x::Array{<:Real,1}, n::Integer)::Set{Integer}
    maxes = Set{Integer}()
    min = findmin(x)[1]
    x_ = x
    for i in range(1, n)
        max_idx = findmax(x_)[2]
        push!(maxes, max_idx)
        x_[max_idx] = min - i
    end
    return maxes
end


function get_default_tightening_options(optimizer)::Dict
    # optimizer_type_name = string(typeof(optimizer()))
    # if optimizer_type_name == "Gurobi.Optimizer"
    #     return Dict("OutputFlag" => 0, "TimeLimit" => 20)
    # elseif optimizer_type_name == "Cbc.Optimizer"
    #     return Dict("logLevel" => 0, "seconds" => 20)
    # else
    #     return Dict()
    # end

    return Dict("OutputFlag" => 0, "TimeLimit" => 20)
end

"""
$(SIGNATURES)

Finds the perturbed image closest to `input` such that the network described by `nn`
classifies the perturbed image in one of the categories identified by the
indexes in `target_selection`.

`optimizer` specifies the optimizer used to solve the MIP problem once it has been built.

The output dictionary has keys `:Model, :PerturbationFamily, :TargetIndexes, :SolveStatus,
:Perturbation, :PerturbedInput, :Output`.
See the [tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/03_interpreting_the_output_of_find_adversarial_example.ipynb)
on what individual dictionary entries correspond to.

*Formal Definition*: If there are a total of `n` categories, the (perturbed) output vector
`y=d[:Output]=d[:PerturbedInput] |> nn` has length `n`.
We guarantee that `y[j] - y[i] ≥ 0` for some `j ∈ target_selection` and for all 
`i ∉ target_selection`.

# Named Arguments:
+ `multilabel::Integer`: The number of labels an input can be classified to. In a regular (not multilabel) 
    classifier its equal to 1.
+ `invert_target_selection::Bool`: Defaults to `false`. If `true`, sets `target_selection` to
    be its complement.
+ `pp::PerturbationFamily`: Defaults to `UnrestrictedPerturbationFamily()`. Determines
    the family of perturbations over which we are searching for adversarial examples.
+ `norm_order::Real`: Defaults to `1`. Determines the distance norm used to determine the
    distance from the perturbed image to the original. Supported options are `1`, `Inf`
    and `2` (if the `optimizer` used can solve MIQPs.)
+ `tightening_algorithm::MIPVerify.TighteningAlgorithm`: Defaults to `mip`. Determines how we
    determine the upper and lower bounds on input to each nonlinear unit.
    Allowed options are `interval_arithmetic`, `lp`, `mip`.
    (1) `interval_arithmetic` looks at the bounds on the output to the previous layer.
    (2) `lp` solves an `lp` corresponding to the `mip` formulation, but with any integer constraints
         relaxed.
    (3) `mip` solves the full `mip` formulation.
+ `tightening_options`: Solver-specific options passed to optimizer when used to determine upper and
    lower bounds for input to nonlinear units. Note that these are only used if the 
    `tightening_algorithm` is `lp` or `mip` (no solver is used when `interval_arithmetic` is used
    to compute the bounds). Defaults for Gurobi and Cbc to a time limit of 20s per solve, 
    with output suppressed.
+ `solve_if_predicted_in_targeted`: Defaults to `true`. The prediction that `nn` makes for the 
    unperturbed `input` can be determined efficiently. If the predicted index is one of the indexes 
    in `target_selection`, we can skip the relatively costly process of building the model for the 
    MIP problem since we already have an "adversarial example" --- namely, the input itself. We 
    continue build the model and solve the (trivial) MIP problem if and only if 
    `solve_if_predicted_in_targeted` is `true`.
"""


function find_adversarial_example(
    nn::NeuralNet,
    input::Array{<:Real},
    target_selection::Union{Integer,Array{<:Integer,1}},
    optimizer,
    main_solve_options::Dict;
    epsilons::AbstractArray{Float64},
    invert_target_selection::Bool = false,
    pp::PerturbationFamily = UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    adversarial_example_objective::AdversarialExampleObjective = worst,
    tightening_algorithm::TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = get_default_tightening_options(optimizer),
    solve_if_predicted_in_targeted = true,
    multilabel::Integer = 1,
    img_idx::String = "",
    optimal = "max",
    limit_time = Inf,
    swap_consts_buf,
    d::Dict
)::Dict

    global finished

    total_time = @elapsed begin

        # Calculate predicted index/es
        predicted_output = input |> nn
        num_possible_indexes = length(predicted_output)

        if multilabel > num_possible_indexes
            error("k must not be larger than the number of possible labels.")
        end

        if multilabel == 1
            predicted_index = predicted_output |> get_max_index
            d[:PredictedIndex] = predicted_index
        else
            predicted_indexes = get_n_maxes_indexes(predicted_output, multilabel)
            d[:PredictedIndexes] = predicted_indexes
        end

        # Set target indexes
        d[:TargetIndexes] = get_target_indexes(
            target_selection,
            num_possible_indexes,
            invert_target_selection = invert_target_selection,
        )
        if multilabel == 1
            notice(
            MIPVerifyMulti.LOGGER,
            "Attempting to find adversarial example. Neural net predicted label is $(predicted_index), target labels are $(d[:TargetIndexes])",
        )
        else
            notice(
            MIPVerifyMulti.LOGGER,
            "Attempting to find adversarial example. Neural net predicted labels are $(collect(predicted_indexes)), target labels are $(d[:TargetIndexes])",
        )
        end
        
        if multilabel == 1
            call_opt = (!(d[:PredictedIndex] in d[:TargetIndexes]) || solve_if_predicted_in_targeted)
        else
            call_opt = true
            for predicted_idx in d[:PredictedIndexes]
                call_opt = call_opt && predicted_idx in d[:TargetIndexes]
            end
            call_opt = !(call_opt) || solve_if_predicted_in_targeted
        end

        # Only call optimizer if predicted index is not found among target indexes.
        if call_opt
            if d[:compute_bounds]
                merge!(d, get_model(nn, input, pp, optimizer, tightening_options, tightening_algorithm, epsilons, img_idx, d[:sub], d[:sub] ? d[:mini_eps] : -1))
                d[:compute_bounds] = false
                d[:Bounds] = IdDict()
                for x_i in d[:Output]
                    d[:Bounds][x_i] = Dict("ub" => Inf, "lb" => -Inf)
                end
            else
                # set some attributes back to default
                # tightening_options[:"BestObjStop"] = -Inf
                # tightening_options[:"MIPGap"] = 1e-4
                set_optimizer(d[:Model], optimizer_with_attributes(optimizer, tightening_options...))
                # set_optimizer_attribute(d[:Model], "BestObjStop", -Inf)
                # set_optimizer_attribute(d[:Model], "MIPGap", 1e-4)
                # set_optimizer_attributes(d[:Model], tightening_options...)
                # set_optimizer_attribute(m, "Presolve", -1)
                # set_optimizer(d[:Model], optimizer_with_attributes(optimizer, tightening_options...))
            end

            add_constraints_buffer(swap_consts_buf, d[:Output])
            m = d[:Model]
            
            if adversarial_example_objective == closest
                set_max_indexes(m, d[:Output], d[:TargetIndexes], multilabel=multilabel)

                # Set perturbation objective
                # NOTE (vtjeng): It is important to set the objective immediately before we carry
                # out the solve. Functions like `set_max_indexes` can modify the objective.
                
                d_p = d[:Perturbation]
                @objective(m, Min, get_norm(norm_order, d_p))
                set_optimizer(m, optimizer)
                set_optimizer_attributes(m, main_solve_options...)
                optimize!(m)
                d[:SolveStatus] = JuMP.termination_status(m)
                d[:SolveTime] = JuMP.solve_time(m)
                d[:BestObjective] = JuMP.objective_value(m)
                
            elseif adversarial_example_objective == worst
                extra_vars = []
                extra_consts = []

                nontarget_indexes = filter((x) -> x ∉ d[:TargetIndexes], 1:num_possible_indexes)

                if optimal == "min"
                    (kth_max_target_var, nontarget_vars, vars1, consts1, d) =
                    get_vars_for_kth_max_index(d[:Output], d[:TargetIndexes], multilabel, d)
                    (opt_nontarget_var, opt_nontarget_arr, filtered_indexes, vars2, consts2, d) = maximum(nontarget_vars, nontarget_indexes, d)
                elseif optimal == "max"
                    (kth_max_target_var, nontarget_vars, vars1, consts1, d) =
                    get_vars_for_kth_max_index(d[:Output], d[:TargetIndexes], multilabel-length(nontarget_indexes)+1, d)
                    (opt_nontarget_var, opt_nontarget_arr, filtered_indexes, vars2, consts2, d) = minimum(nontarget_vars, nontarget_indexes, d)
                else
                    error("optimal must be either \"max\" or \"min\"")
                end

                # if optimal == "min"
                #     (opt_nontarget_var, vars2, consts2) = maximum_ge(nontarget_vars)
                # elseif optimal == "max"
                #     (opt_nontarget_var, vars2, consts2) = minimum_ge(nontarget_vars)
                # else
                #     error("optimal must be either \"max\" or \"min\"")
                # end

                append!(extra_vars, vars1)
                append!(extra_consts, consts1)
                append!(extra_vars, vars2)
                append!(extra_consts, consts2)
                
                # Introduce an additional variable since Gurobi ignores constant terms in objective, 
                # but we explicitly need these if we want to stop early based on the value of the 
                # objective (not simply whether or not it is maximized).
                # See discussion in https://github.com/jump-dev/Gurobi.jl/issues/111 for more 
                # details.

                # THIS IS EQUAL TO THE CONSTRAINT:
                # MIN{ confidence(x')[unperturbed_class] - 2ndMAx{confidence(x')[other_class] | other_class != unperturbed_class} | x' in I(x+-eps)}
                
                v_obj = @variable(m)
                push!(extra_consts, @constraint(m, v_obj == kth_max_target_var - opt_nontarget_var))

                if optimal == "max"
                    push!(extra_consts, @constraint(m, v_obj >= 0))
                    @objective(m, Max, v_obj)
                elseif optimal == "min"
                    push!(extra_consts, @constraint(m, v_obj <= 0))
                    @objective(m, Min, v_obj)
                else
                    error("optimal must be either \"max\" or \"min\"")
                end
                
                # set_optimizer_attributes(m, main_solve_options...)
                set_optimizer(m, optimizer_with_attributes(optimizer, main_solve_options...))
                set_time_limit_sec(m, limit_time)
                # set_optimizer(m, optimizer)
                # set_optimizer_attributes(m, main_solve_options...)
                

                # set_optimizer_attribute(m, "BestObjStop", 0)  # for early stopping (a non robust solution (adv example) is found)
                # set_optimizer_attribute(m, "MIPGap", 0.99)  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                # set_optimizer_attribute(m, "Presolve", 0)
                
                optimize!(m)
                d[:SolveStatus] = JuMP.termination_status(m)
                d[:SolveTime] = JuMP.solve_time(m)
                # d[:BestObjective] = JuMP.objective_value(m)

                i = get_first_true_index(opt_nontarget_arr)
                d[:OptNonTarget] = nontarget_indexes[filtered_indexes][i]

                GRBreset(Ref(m), 0)
                for var in extra_vars
                    delete(m, var)
                end
                for constr in extra_consts
                    delete(m, constr)
                end
                
            else
                error("Unknown adversarial_example_objective $adversarial_example_objective")
            end
        end
    end

    return d 

end

function relation_feasibility(
    nn::NeuralNet,
    input::Array{<:Real},
    optimizer,
    main_solve_options::Dict;
    epsilons::AbstractArray{Float64},
    l1::Integer,
    l2::Integer,
    pp::PerturbationFamily = UnrestrictedPerturbationFamily(),
    tightening_algorithm::TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = get_default_tightening_options(optimizer),
    img_idx::String = "",
    limit_time = Inf,
    d::Dict
)
    
    notice(
        MIPVerifyMulti.LOGGER,
        "Attempting to find example such that label $l1 <= label $l2",
    )

    if d[:compute_bounds]
        merge!(d, get_model(nn, input, pp, optimizer, tightening_options, tightening_algorithm, epsilons, img_idx))
        d[:compute_bounds] = false
        d[:Bounds] = IdDict()
        for x_i in d[:Output]
            d[:Bounds][x_i] = Dict("ub" => Inf, "lb" => -Inf)
        end
    end
    m = d[:Model]

    extra_consts = []

    xs = d[:Output]
    push!(extra_consts, @constraint(m, xs[l1] <= xs[l2]))

    v_obj = @variable(m)
    push!(extra_consts, @constraint(m, v_obj == xs[l1] - xs[l2]))
    @objective(m, Min, v_obj)

    set_optimizer(m, optimizer_with_attributes(optimizer, main_solve_options...))
    set_time_limit_sec(m, limit_time)

    # set_optimizer(m, optimizer)
    # set_optimizer_attributes(m, main_solve_options...)

    # if limit_time != Inf
    #     set_time_limit_sec(m, limit_time)
    # end
    # set_optimizer_attribute(m, "BestObjStop", 0)  # for early stopping (a non robust solution (adv example) is found)
    # set_optimizer_attribute(m, "MIPGap", 0.99)  # for early stopping (lower and upper bounds of optimal solution have the same sign)
    # # set_optimizer_attribute(m, "Presolve", 0)
    
    optimize!(m)
    
    d[:SolveStatus] = JuMP.termination_status(m)
    print("\nStatus: $(d[:SolveStatus])\n")
    
    d[:SolveTime] = JuMP.solve_time(m)

    GRBreset(Ref(m), 0)
    
    for constr in extra_consts
        delete(m, constr)
    end

    return d
                
end

function get_label(y::Array{<:Real,1}, test_index::Integer)::Int
    return y[test_index]
end

function get_image(x::Array{T,4}, test_index::Integer)::Array{T,4} where {T<:Real}
    return x[test_index:test_index, :, :, :]
end

"""
$(SIGNATURES)

Returns the fraction of items the neural network correctly classifies of the first
`num_samples` of the provided `dataset`. If there are fewer than
`num_samples` items, we use all of the available samples.

# Named Arguments:
+ `nn::NeuralNet`: The parameters of the neural network.
+ `dataset::LabelledDataset`:
+ `num_samples::Integer`: Number of samples to use.
"""
function frac_correct(nn::NeuralNet, dataset::LabelledDataset, num_samples::Integer)::Real

    num_correct = 0.0
    num_samples = min(num_samples, MIPVerifyMulti.num_samples(dataset))
    @showprogress 1 "Computing fraction correct..." for sample_index in 1:num_samples
        input = get_image(dataset.images, sample_index)
        actual_label = get_label(dataset.labels, sample_index)
        predicted_label = (input |> nn |> get_max_index) - 1
        if actual_label == predicted_label
            num_correct += 1
        end
    end
    return num_correct / num_samples
end


function get_norm(norm_order::Real, v::Array{<:Real})
    if norm_order == 1
        return sum(abs.(v))
    elseif norm_order == 2
        return sqrt(sum(v .* v))
    elseif norm_order == Inf
        result = Base.maximum(Iterators.flatten(abs.(v)))
        if result == 0  # dont allow 0 distance
            return Inf
        else
            return result
        end
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

function get_norm(norm_order::Real, v::Array{<:JuMPLinearType,N}) where {N}
    if norm_order == 1
        abs_v = abs_ge.(v)
        return sum(abs_v)
    elseif norm_order == 2
        return sum(v .* v)
    elseif norm_order == Inf
        result = MIPVerifyMulti.maximum_ge(permute_and_flatten(abs_ge.(v), N:-1:1))
        # model = owner_model(result)
        # big_constant = @variable(model)
        # @constraint(model, big_constant*result >= 0.001)
        # @constraint(model, big_constant == 10^(6))
        return result
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

include("batch_processing_helpers.jl")

end
