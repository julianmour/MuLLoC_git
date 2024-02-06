using Serialization
using Decimals
using JLD2
using Gurobi


const big_constant = 10^8

"""
Supertype for types encoding the family of perturbations allowed.
"""
abstract type PerturbationFamily end

struct UnrestrictedPerturbationFamily <: PerturbationFamily end
Base.show(io::IO, pp::UnrestrictedPerturbationFamily) = print(io, "unrestricted")

abstract type RestrictedPerturbationFamily <: PerturbationFamily end

small_const = 0.95

"""
For blurring perturbations, we currently allow colors to "bleed" across color channels -
that is, the value of the output of channel 1 can depend on the input to all channels.
(This is something that is worth reconsidering if we are working on color input).
"""
struct BlurringPerturbationFamily <: RestrictedPerturbationFamily
    blur_kernel_size::NTuple{2}
end
Base.show(io::IO, pp::BlurringPerturbationFamily) =
    print(io, filter(x -> !isspace(x), "blur-$(pp.blur_kernel_size)"))

struct LInfNormBoundedPerturbationFamily <: RestrictedPerturbationFamily
    norm_bound::Real

    function LInfNormBoundedPerturbationFamily(norm_bound::Real)
        @assert(norm_bound > 0, "Norm bound $(norm_bound) should be positive")
        return new(norm_bound)
    end
end
Base.show(io::IO, pp::LInfNormBoundedPerturbationFamily) =
    print(io, "linf-norm-bounded-$(pp.norm_bound)")

function get_model(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily,
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
    epsilons::AbstractArray{Float64},
    img_idx::String="",
    sub=false,
    mini_eps=0.001
)::Dict{Symbol,Any}
    notice(
        MIPVerifyMulti.LOGGER,
        "Determining upper and lower bounds for the input to each non-linear unit.",
    )

    m = Model(optimizer_with_attributes(optimizer, tightening_options...))

    m.ext[:MIPVerifyMulti] = MIPVerifyExt(tightening_algorithm)

    d_common = Dict(
        :Model => m,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    if !sub
        return merge(d_common, get_perturbation_specific_keys(nn, input, pp, m, epsilons, img_idx))
    else
        return merge(d_common, get_perturbation_specific_keys_sub(nn, input, pp, m, epsilons, img_idx, mini_eps))
    end
end

function if_all_nonzeros(mat)
    # mat = mat_[1,1:end,1:end,1]
    # mat is a nxn matrix
    n = 64
    for i in range(1,n)
        for j in range(1,n)
            if mat[1,i,j,1] == 0
                return false
            end
        end
    end
    return true
end


function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    m::Model,
    epsilons::AbstractArray{Float64},
    img_idx::String="",
    one_layer=true,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))

    masks_path = raw"masks"
    i = img_idx
    len = length(epsilons)

    if one_layer
        
        lower_bounds = input .- epsilons[1]
        upper_bounds = input .+ epsilons[1]

    else
        mask1 = load(joinpath(masks_path, "img"*string(i), "mask"*string(i)*"_delta0.png"))
        mask1 = convert(Array{Float64}, mask1)
        padded_shape_ = (1, size(mask1)..., 1)
        mask1 = reshape(mask1, padded_shape_)

        mask_inverted = 1 .- mask1
        lower_bounds = mask_inverted .* input   # 0 for mask region, input value for region outside of mask
        upper_bounds = lower_bounds .+ mask1    # 1 for mask region, input value for region outside of mask
        
        eps = epsilons[1]
        
        mask_input_lower = mask1 .* (input .- eps)
        mask_input_upper = mask1 .* (input .+ eps)
        lower_bounds_tmp = lower_bounds
        lower_bounds = lower_bounds .+ mask_input_lower   # pix-eps for mask region, input value for region outside of mask
        upper_bounds = lower_bounds_tmp .+ mask_input_upper    # pix+eps for mask region, input value for region outside of mask

        # for the rest of the layer_mask

        for l in range(2, len)

            mask1 = load(joinpath(masks_path, "img"*string(i), "mask"*string(i)*"_delta"*string(l-2)*".png"))
            mask1 = convert(Array{Float64}, mask1)
            padded_shape_ = (1, size(mask1)..., 1)
            mask1 = reshape(mask1, padded_shape_)
            
            mask2 = load(joinpath(masks_path, "img"*string(i), "mask"*string(i)*"_delta"*string(l-1)*".png"))
            mask2 = convert(Array{Float64}, mask2)
            padded_shape_ = (1, size(mask2)..., 1)
            mask2 = reshape(mask2, padded_shape_)

            layer_mask = mask2 .- mask1
                    
            lower_bounds = lower_bounds .- (layer_mask.* lower_bounds)   # 0 for layer mask region
            upper_bounds = upper_bounds .- (layer_mask.* upper_bounds) .+ layer_mask    # 1 for layer mask region
            
            eps = epsilons[l]
            
            mask_input_lower = layer_mask .* (input .- eps)
            mask_input_upper = layer_mask .* (input .+ eps)
            lower_bounds = lower_bounds + mask_input_lower   # pix-eps for mask region, input value for region outside of mask
            upper_bounds = upper_bounds .- (layer_mask.* upper_bounds) .+ mask_input_upper    # pix+eps for mask region, input value for region outside of mask

        end

    end

    # input_range_ = deepcopy(input_range)
    # u_l = sum(map(I -> ((min(1, upper_bounds[I]) - max(0, lower_bounds[I])) >= 0 ? 0 : 1), input_range_))
    # print("VALID BOUNDS: ", string(u_l == 0),"\n\n")

    # v_x0 is the input with the perturbation added
    v_x0 = map(I -> @variable(m, lower_bound =  max(0, lower_bounds[I]), upper_bound = min(1, upper_bounds[I])), input_range)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_x0 - input, :Output => v_output)

end

function get_perturbation_specific_keys_sub(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    m::Model,
    epsilons::AbstractArray{Float64},
    img_idx::String="",
    mini_eps=0.001,
    one_layer=true
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))

    if one_layer
        
        lower_bounds = input .+ epsilons[1] .- mini_eps
        upper_bounds = input .+ epsilons[1]

    end

    # input_range_ = deepcopy(input_range)
    # u_l = sum(map(I -> ((min(1, upper_bounds[I]) - max(0, lower_bounds[I])) >= 0 ? 0 : 1), input_range_))
    # print("VALID BOUNDS: ", string(u_l == 0),"\n\n")

    # v_x0 is the input with the perturbation added
    v_x0 = map(I -> @variable(m, lower_bound =  min(1, max(0, lower_bounds[I])), upper_bound = min(1, upper_bounds[I])), input_range)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_x0 - input, :Output => v_output)

end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::BlurringPerturbationFamily,
    m::Model,
    delta::Integer = 0,
    img_idx::String="",
)::Dict{Symbol,Any}

    input_size = size(input)
    num_channels = size(input)[4]
    filter_size = (pp.blur_kernel_size..., num_channels, num_channels)

    v_f = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(filter_size))
    @constraint(m, sum(v_f) == num_channels)
    v_x0 = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(input_size))
    @constraint(m, v_x0 .== input |> Conv2d(v_f))

    v_output = v_x0 |> nn

    return Dict(
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :BlurKernel => v_f,
    )
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
    m::Model,
    delta::Integer = 0,
    img_idx::String="",
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_e = map(
        _ -> @variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound),
        input_range,
    )
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - pp.norm_bound),
            upper_bound = min(1, input[i] + pp.norm_bound)
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end

struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
end
