############## Helper Funcs ################

function ToSoftmax(x)
    e_elements = zeros(size(x))
    for (i, element) in enumerate(x)
        e_elements[i] = ℯ^element
    end
    y = sum(e_elements)
    softmax = e_elements ./ y
    return softmax
end

function count_different_entries(matrix1, matrix2)
    if size(matrix1) != size(matrix2)
        error("Matrices must have the same dimensions")
    end
    
    # Compare corresponding elements and count differences
    num_diff_entries = sum(matrix1 .!= matrix2)
    return num_diff_entries
end

function initMatrixes(m)
    gtm = zeros(m, m) .- 1      # always >= matrix
    ltm = zeros(m, m) .- 1      # always < matrix
    for class in 1:m
        gtm[class,class] = 1
        ltm[class,class] = 0
    end
    return (gtm, ltm)
end

function updateMatrixes(scores, gtm, ltm)
    gtm_old = copy(gtm)
    ltm_old = copy(ltm)
    for (class1, score1) in enumerate(scores)
        for (class2, score2) in enumerate(scores[class1+1:end])
            if score1 < score2
                gtm[class1, class2] = 0
            else
                ltm[class1, class2] = 0
            end
        end
    end
    gtm_diff = count_different_entries(gtm, gtm_old)
    ltm_diff = count_different_entries(ltm, ltm_old)
    return (gtm, ltm, gtm_diff, ltm_diff)
end

function random_matrix(n1, n2, min_val, max_val, extreme=false, c=1)
    if !extreme
        rand_vals = (c==1 ? rand(n1, n2) : rand(c, n1, n2))  # Generate random values between 0 and 1
    else
        rand_vals = (c==1 ? rand(Bool, n1, n2) : rand(Bool, c, n1, n2))  # Generate random values of 0 or 1
    end
    scaled_vals = min_val .+ rand_vals .* (max_val - min_val)  # Scale to desired range
    return scaled_vals
end

function cut_vals(val)
    return val > 1 ? 1 : (val < 0 ? 0 : val)
end

function sample_img(img, eps, extreme=false)
    if length(size(img)) == 2
        (n1, n2) = size(img)
        pert = convert(Array{Float64}, random_matrix(n1, n2, -eps, eps, extreme))
    elseif length(size(img)) == 3
        (c, n1, n2) = size(img)
        pert = convert(Array{Float64}, random_matrix(n1, n2, -eps, eps, extreme, c))
    else
        error("Unrecognized image number of dimensions.")
    end
    return cut_vals.(convert(Array{Float64}, img) .+ pert)

end

function translate_classification(classification, k)
    return [translate_class(x, k) for x in classification]
end

function translate_class(c, k)
    if c == 0
        return "ALWAYS top-$k"
    elseif c == 1
        return "ALWAYS not top-$k"
    elseif c == 2
        return "top-$k OR not top-$k"
    else
        return "--unknown--"    # shouldnt reach this
    end
end

function get_labels_lowest_set(scores, thresh)
    sum = 0
    for (idx, s) in enumerate(reverse(scores))
        sum += s
        if sum >= thresh
            return idx - 1
        end
    end
    return length(scores)
end

function init_Ds(mini_eps)
    d = Dict()
    d[:compute_bounds] = true
    d[:sub] = false

    sub_d = Dict()
    sub_d[:compute_bounds] = true
    sub_d[:sub] = true
    sub_d[:mini_eps] = mini_eps

    return (d, sub_d)
end

function prepare_runs()
    runs = []
#     for eps in [0.0001, 0.2]
    for eps in [0.2]
#         for k in [2, 3]
        for k in [3]
            push!(runs, Dict("dataset" => "dmnist", "model" => "dmnist_noDef_x2", "eps" => eps, "k" => k))
        end
    end

    for eps in [0.0001, 0.0005]
        for k in [2, 3]
            push!(runs, Dict("dataset" => "tmnist", "model" => "tmnist_noDef_x2", "eps" => eps, "k" => k))
        end
    end
    push!(runs, Dict("dataset" => "tmnist", "model" => "tmnist_noDef_x2", "eps" => 0.01, "k" => 2))
    push!(runs, Dict("dataset" => "tmnist", "model" => "tmnist_noDef_x2", "eps" => 0.0002, "k" => 3))

    for eps in [0.0001, 0.0002, 0.0005, 0.001]
        for k in [2, 3]
            push!(runs, Dict("dataset" => "tmnist", "model" => "tmnist_noDef_x3", "eps" => eps, "k" => k))
        end
    end

    for eps in [0.0001, 0.0005]
        for k in [3, 4]
            push!(runs, Dict("dataset" => "pascal-voc", "model" => "pascal-voc_noDef_x3", "eps" => eps, "k" => k))
        end
    end
    push!(runs, Dict("dataset" => "pascal-voc", "model" => "pascal-voc_noDef_x3", "eps" => 0.001, "k" => 3))

    push!(runs, Dict("dataset" => "pascal-voc", "model" => "pascal-voc_noDef_x3", "eps" => 0.0001, "k" => "gt"))
    push!(runs, Dict("dataset" => "pascal-voc", "model" => "pascal-voc_noDef_x3", "eps" => 0.0005, "k" => "gt"))

    return runs
end

