export get_matrix_params, get_conv_params, get_example_network_params

"""

Helper function to import the parameters for a layer carrying out matrix multiplication
    (e.g. fully connected layer / softmax layer) from `param_dict` as a
    [`Linear`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{2, Int}`: Tuple of length 2 corresponding to the expected size
   of the weights of the layer.

"""
function flip_90(matrix)
    new_matrix = zeros(size(matrix)[2], size(matrix)[1])
    for i in range(1,size(matrix)[1])
        for j in range(1,size(matrix)[2])
            new_matrix[j, size(matrix)[1]-i+1] = matrix[i, j]
        end
    end
    return new_matrix
end

function flip_90_each(matrix)
    new_matrix = matrix
    for i in range(1,size(new_matrix)[1])
        for j in range(1,size(new_matrix)[2])
            new_matrix[i, j, 1:end, 1:end] = flip_90(new_matrix[i, j, 1:end, 1:end])
        end
    end
    return new_matrix
end


function get_matrix_params(
    param_dict,
    layer_name::String,
    expected_size::NTuple{2,<:Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
)::Linear

    params = Linear(
        permutedims(param_dict["$layer_name.$matrix_name"],(2,1)),
        # dropdims(param_dict["$layer_name.$bias_name"], dims = 1),
        param_dict["$layer_name.$bias_name"],

    )

    check_size(params, expected_size)

    return params
end

"""
Helper function to import the parameters for a convolution layer from `param_dict` as a
    [`Conv2d`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{4, Int}`: Tuple of length 4 corresponding to the expected size
    of the weights of the layer.
    
"""

function get_conv_params(
    param_dict,
    layer_name::String,
    expected_size::NTuple{4,<:Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1,
    padding::Padding = SamePadding(),
)::Conv2d

    params = Conv2d(
        permutedims(param_dict["$layer_name.$matrix_name"],(3, 4, 2, 1)),   # permutation is since we are using a pytorch model (instead of TensorFlow)
        # dropdims(param_dict["$layer_name.$bias_name"], dims = 1),
        param_dict["$layer_name.$bias_name"],
        expected_stride,
        padding,
    )

    check_size(params, expected_size)

    return params
end
