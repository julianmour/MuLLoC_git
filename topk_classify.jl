using Revise
using Gurobi
using PyCall
using JLD2
using JuMP
using NPZ
using Statistics
using Images, FileIO
using Base.Threads
using Dates
using CSV
using DataFrames
using StatsBase
using Base.Filesystem


include("MIPVerifyMulti.jl")
include("helperFuncs.jl")

using .MIPVerifyMulti

torch = pyimport("torch")
pickle = pyimport("pickle")
np = pyimport("numpy")

dataset = "tmnist"
# model_names = ["dmnist_noDef", "dmnist_L0Def", "dmnist_LInfDef"]
model_names = ["tmnist_noDef_x2"]
# ks = [2, 3]
ks = [2]

# epsilons = [0.003]
epsilons = [0.01]


skip_sampling = false
const DEFAULT_RANDOM_SAMPLING_DURATION = 1
const DEFAULT_CORNERS_SAMPLING_DURATION = 1

skip_attacking = false
skip_mini_milps = false
const DEFAULT_MINI_MILP_TIME_LIMIT = 10

skip_super_milps = false
const DEFAULT_HEURISTICS_PARAMETER = 0.05

skip_sub_milps = true
mini_eps = 0.001


for model_name in model_names

    ############## Model loading ###############

    if dataset == "dmnist"
        if model_name == "dmnist_LInfDef_x1k"
            param_dict = torch.load(joinpath("models", "dmnist_model394_4k_params_LInfDef"), map_location=torch.device("cpu"))
        elseif model_name == "dmnist_L0Def_x1k"
            param_dict = torch.load(joinpath("models", "dmnist_model421_4k_params_L0Def"), map_location=torch.device("cpu"))
        elseif model_name == "dmnist_noDef_x1k"
            param_dict = torch.load(joinpath("models", "dmnist_model381_4k_params_noDef"), map_location=torch.device("cpu"))
        elseif model_name == "dmnist_noDef_x3"
            param_dict = torch.load(joinpath("models", "dmnist_model76_x3_noDef"), map_location=torch.device("cpu"))
        elseif model_name == "dmnist_noDef_x2"
            param_dict = torch.load(joinpath("models", "dmnist_model262_x2_noDef"), map_location=torch.device("cpu"))
        else
            error("dataset $dataset doesnt match model name $model_name (or model does not exist)")
        end

        for (p_idx, value) in param_dict
            param_dict[p_idx] = value[:numpy]()
        end
        
        if model_name == "dmnist_LInfDef_x1k" || model_name == "dmnist_L0Def_x1k" || model_name == "dmnist_noDef_x1k"
            conv1 = MIPVerifyMulti.get_conv_params(param_dict, "conv1", (3, 3, 1, 8), expected_stride=2, padding=1)
            conv2 = MIPVerifyMulti.get_conv_params(param_dict, "conv2", (3, 3, 8, 16), expected_stride=2, padding=1)
        
            fc1 = MIPVerifyMulti.get_matrix_params(param_dict, "fc1", (256, 10))
            
            # The dmnsit NN
            model = MIPVerifyMulti.Sequential([
                    conv1,
                    MIPVerifyMulti.ReLU(),
                    conv2,
                    MIPVerifyMulti.ReLU(),
                    MIPVerifyMulti.MaxPool((1, 4, 4, 1)),
                    MIPVerifyMulti.Flatten(4, [1,3,2,4]),   # permute with [1,3,2,4] if using a pytorch model
                    fc1,
                ], "model")

        elseif model_name == "dmnist_noDef_x3"
            conv1 = MIPVerifyMulti.get_conv_params(param_dict, "conv1", (4, 4, 1, 8), expected_stride=2, padding=1)
            conv2 = MIPVerifyMulti.get_conv_params(param_dict, "conv2", (4, 4, 8, 16), expected_stride=4, padding=1)
            conv3 = MIPVerifyMulti.get_conv_params(param_dict, "conv3", (4, 4, 16, 32), expected_stride=2, padding=1)
        
            fc1 = MIPVerifyMulti.get_matrix_params(param_dict, "fc1", (512, 10))
            
            # The dmnsit NN
            model = MIPVerifyMulti.Sequential([
                    conv1,
                    MIPVerifyMulti.ReLU(),
                    conv2,
                    MIPVerifyMulti.ReLU(),
                    conv3,
                    MIPVerifyMulti.ReLU(),
                    MIPVerifyMulti.Flatten(4, [1,3,2,4]),   # permute with [1,3,2,4] if using a pytorch model
                    fc1,
                ], "model")

        elseif model_name == "dmnist_noDef_x2"
            conv1 = MIPVerifyMulti.get_conv_params(param_dict, "conv1", (4, 4, 1, 16), expected_stride=4, padding=1)
            conv2 = MIPVerifyMulti.get_conv_params(param_dict, "conv2", (4, 4, 16, 32), expected_stride=4, padding=1)
        
            fc1 = MIPVerifyMulti.get_matrix_params(param_dict, "fc1", (512, 10))
            
            # The dmnsit NN
            model = MIPVerifyMulti.Sequential([
                    conv1,
                    MIPVerifyMulti.ReLU(),
                    conv2,
                    MIPVerifyMulti.ReLU(),
                    MIPVerifyMulti.Flatten(4, [1,3,2,4]),   # permute with [1,3,2,4] if using a pytorch model
                    fc1,
                ], "model")

        end
    
    elseif dataset == "tmnist"
        if model_name == "tmnist_noDef_x3"
            param_dict = torch.load(joinpath("models", "tmnist_model23_x3_noDef"), map_location=torch.device("cpu"))
            for (p_idx, value) in param_dict
                param_dict[p_idx] = value[:numpy]()
            end
        
            conv1 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv1", (4, 4, 1, 8), expected_stride=2, padding=1)
            conv2 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv2", (3, 3, 8, 16), expected_stride=3, padding=1)
            conv3 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv3", (4, 4, 16, 32), expected_stride=2, padding=1)
            
            fc1 = MIPVerifyMulti.get_matrix_params(param_dict, "module.fc1", (1568, 10))
        
        
            # The tmnsit NN
            model = MIPVerifyMulti.Sequential([
                    conv1,
                    MIPVerifyMulti.ReLU(),
                    conv2,
                    MIPVerifyMulti.ReLU(),
                    conv3,
                    MIPVerifyMulti.ReLU(),
                    MIPVerifyMulti.Flatten(4, [1,3,2,4]),   # permute with [1,3,2,4] if using a pytorch model
                    fc1,
                ], "model")

        elseif model_name == "tmnist_noDef_x2"
            param_dict = torch.load(joinpath("models", "tmnist_model111_x2_noDef"), map_location=torch.device("cpu"))
            for (p_idx, value) in param_dict
                param_dict[p_idx] = value[:numpy]()
            end
        
            conv1 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv1", (4, 4, 1, 16), expected_stride=4, padding=1)
            conv2 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv2", (3, 3, 16, 32), expected_stride=3, padding=1)
            
            fc1 = MIPVerifyMulti.get_matrix_params(param_dict, "module.fc1", (1568, 10))
        
        
            # The tmnsit NN
            model = MIPVerifyMulti.Sequential([
                    conv1,
                    MIPVerifyMulti.ReLU(),
                    conv2,
                    MIPVerifyMulti.ReLU(),
                    MIPVerifyMulti.Flatten(4, [1,3,2,4]),   # permute with [1,3,2,4] if using a pytorch model
                    fc1,
                ], "model")
        else
            error("dataset $dataset doesnt match model name $model_name (or model does not exist)")
        end
    
    elseif dataset == "pascal-voc"
        if model_name == "pascal-voc_noDef_x3"
            param_dict = torch.load(joinpath("models", "pascal-voc_model272_x3"), map_location=torch.device("cpu"))
            for (p_idx, value) in param_dict
                param_dict[p_idx] = value[:numpy]()
            end
        
            conv1 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv1", (5, 5, 3, 8), expected_stride=5, padding=2)
            conv2 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv2", (4, 4, 8, 16), expected_stride=4, padding=1)
            conv3 = MIPVerifyMulti.get_conv_params(param_dict, "module.conv3", (3, 3, 16, 32), expected_stride=3, padding=1)
            
            fc1 = MIPVerifyMulti.get_matrix_params(param_dict, "module.fc1", (800, 20))
        
        
            # The tmnsit NN
            model = MIPVerifyMulti.Sequential([
                    conv1,
                    MIPVerifyMulti.ReLU(),
                    conv2,
                    MIPVerifyMulti.ReLU(),
                    conv3,
                    MIPVerifyMulti.ReLU(),
                    MIPVerifyMulti.Flatten(4, [1,3,2,4]),   # permute with [1,3,2,4] if using a pytorch model
                    fc1,
                ], "model")

        else
            error("dataset $dataset doesnt match model name $model_name (or model does not exist)")
        end
    else
        error("dataset $dataset doesnt exist")
    end


    ############## Images loading ############

    samples_path = dataset*"_samples"
    img_names = [splitext(name)[1] for name in readdir(samples_path)]
    img_files = readdir(samples_path)
    complex = split(model_name, '_')[3]

    for k_ in ks

        for eps in epsilons

            @assert (k_=="gt" || k_==1 || k_==2 || k_==3 || k_==4 || k_==5 || k_==6 || k_==7 || k_==8)
            @assert (eps>= 0 && eps<=1) "epsilon must be between 0 and 1"
            
            skip_sampling_txt = skip_sampling ? "_skipSampling" : ""
            skip_attacking_txt = skip_attacking ? "_skipAttacking" : ""
            skip_milp_relations_txt = skip_mini_milps ? "_skipMiniMilps" : ""
            # skip_sub_milps_txt = skip_sub_milps ? "_skipSubMilps" : ""
            skip_super_milps_txt = skip_super_milps ? "_skipSuperMilps" : ""
            results_folder = joinpath("results", dataset, model_name, "k_$k_", "eps_$eps$skip_sampling_txt$skip_attacking_txt$skip_milp_relations_txt$skip_super_milps_txt")

            if !isdir(results_folder)
                mkpath(results_folder)
            end

            random_sampling_duration = DEFAULT_RANDOM_SAMPLING_DURATION
            corners_sampling_duration = DEFAULT_CORNERS_SAMPLING_DURATION
            
            start_from_img_idx = 1
            till_img_idx = 30
            
            # skip = [1,3,5,7,9,11,13,15,17,19]
            skip = []

            mini_milp_time_limit = DEFAULT_MINI_MILP_TIME_LIMIT


            for (i_idx, i) in enumerate(img_names)

                mini_milps_num = 0
                super_milps_num = 0
                regular_milps_num = 0

                if i_idx < start_from_img_idx
                    continue
                end

                if i_idx in skip
                    continue
                end

                output_path = joinpath("$results_folder", "$i")
                if !isdir(output_path)
                    mkdir(output_path)
                end

                k = k_
                if k_ == "gt"
                    if dataset == "dmnist"
                        k = 2
                    elseif dataset == "tmnist"
                        k = 3
                    elseif dataset == "pascal-voc"
                        k = length(split(split(i, '_')[2], '-'))
                    end

                    results_folder_check = joinpath("results", dataset, model_name, "k_$k", "eps_$eps$skip_sampling_txt$skip_attacking_txt$skip_milp_relations_txt$skip_super_milps_txt")
                    output_path_check = joinpath("$results_folder_check", "$i")
                    if isfile(joinpath("$output_path_check", "classification_summary.csv")) && isfile(joinpath("$output_path_check", "exec_summary.csv")) && isfile(joinpath("$output_path_check", "milp_summary.csv")) && isfile(joinpath("$output_path_check", "params_summary.csv"))
                        cp("$output_path_check", "$output_path", force=true)
                        print("\ncopied results from existing directory...\n")
                        continue
                    end
                end

                start_time = now()

                print("\n\n******************************************************************************************")
                print("\n************** MODEL $model_name | IMAGE $i | EPS = $eps | K = $k **************\n")
                print("******************************************************************************************\n\n")
                img_ = load(joinpath(samples_path, string(i)*(dataset=="pascal-voc" ? ".jpeg" : ".png")))
                if dataset=="pascal-voc"
                    img_ = imresize(img_, 300, 300)
                    img_ = float.(channelview(img_))
                    img_ = permutedims(img_, (2, 3, 1))
                    img_org = img_
                    padded_shape_ = (1, size(img_)...)
                    img_ = reshape(img_, padded_shape_)
                else
                    img_ = convert(Array{Float64}, img_)
                    img_org = img_
                    padded_shape_ = (1, size(img_)..., 1)
                    img_ = reshape(img_, padded_shape_)
                end
                result_ = ToSoftmax(model(img_))
                print("image ", string(i), ":\n")
                print("NN result: " , result_, "\n")
                ranking = reverse(sortperm(result_))
                print("Classes ranking: ", ranking, "\n\n")
                m = length(result_)

                (gtm, ltm) = initMatrixes(m)    # gtm: always >= matrix, ltm: always < matrix
                classes_classification = zeros(m) .- 1    # -1: unfilled, 0: always topk, 1: always not in topk, 2: might be in both
                classes_classification_time = zeros(m)    # time till classified
                classes_classification_part = fill("", m)    # part in which the label got classified
                topk = ranking[1:k]
                not_topk = ranking[k+1:end]
                overall_updated = 0

                (d, sub_d) = init_Ds(mini_eps)

                print("**************** PRUNING RELATIONS USING ORIGINAL IMAGE ****************\n\n")
                
                # update matrixes according to image scores
                (gtm, ltm, gtm_diff, ltm_diff) = updateMatrixes(result_, gtm, ltm)
                if gtm_diff + ltm_diff > 0
                    overall_updated += (gtm_diff + ltm_diff)
                    print("# updated entries = ", gtm_diff + ltm_diff, "\n")
                    print("# overall updated entries = ", overall_updated, "\n\n")
                end


                print("**************** PRUNING RELATIONS USING SAMPLED IMAGES ****************\n\n")
                
                random_sampling_classified = 0
                corner_sampling_classified = 0
                average_ranking = ranking

                if skip_sampling == false

                    labels_rankings_avg = zeros(m)

                    # update matrixes according to sampled images in neighborhood
                    si_idx = 1
                    keep_generating = true

                    stop_sampling_timer = now()
                    while keep_generating

                        if all(x -> x != -1.0, classes_classification)  # all labels classified
                            break
                        end

                        # print(updated_counter, "\n")

                        sampled_img = sample_img(img_org, eps)
                        sampled_img = convert(Array{Float64}, sampled_img)
                        if dataset == "pascal-voc"
                            padded_shape_ = (1, size(sampled_img)...)
                        else
                            padded_shape_ = (1, size(sampled_img)..., 1)
                        end
                        sampled_img = reshape(sampled_img, padded_shape_)
                        sampled_img_result_ = ToSoftmax(model(sampled_img))
                        sampled_ranking = reverse(sortperm(sampled_img_result_))

                        # update rankings statistics
                        for (ranking_, label) in enumerate(sampled_ranking)
                            # push!(labels_rankings[label], ranking_)
                            labels_rankings_avg[label] += ranking_
                        end

                        # update matrixes according to image scores
                        (gtm, ltm, gtm_diff, ltm_diff) = updateMatrixes(sampled_img_result_, gtm, ltm)


                        if gtm_diff + ltm_diff > 0
                            overall_updated += (gtm_diff + ltm_diff)
                            print("sampled image ", string(si_idx), ":\n")
                            print("# updated entries = ", gtm_diff + ltm_diff, "\n")
                            print("# overall updated entries = ", overall_updated, "\n")
                            print("NN result: " , sampled_img_result_, "\n")
                            print("Classes ranking: ", sampled_ranking, "\n\n")
                            stop_sampling_timer = now()
                        end

                        sampled_topk = sampled_ranking[1:k]
                        for top_class in topk
                            if !(top_class in sampled_topk)
                                prev = classes_classification[top_class]
                                classes_classification[top_class] = 2
                                if prev != 2
                                    if gtm_diff + ltm_diff == 0
                                        print("sampled image ", string(si_idx), ":\n")
                                        print("new TOPK class classified: ", top_class,"\n")
                                        print("NN result: " , sampled_img_result_, "\n")
                                        print("Classes ranking: ", sampled_ranking, "\n\n")
                                    else
                                        print("new TOPK class classified: ", top_class,"\n\n")
                                    end
                                    time = now() - start_time
                                    classes_classification_time[top_class] = Float64(time.value)
                                    classes_classification_part[top_class] = "random sampling"
                                    random_sampling_classified += 1
                                    stop_sampling_timer = now()
                                end
                            end
                        end
                        for not_top_class in not_topk
                            if not_top_class in sampled_topk
                                prev = classes_classification[not_top_class]
                                classes_classification[not_top_class] = 2
                                if prev != 2
                                    if gtm_diff + ltm_diff == 0
                                        print("sampled image ", string(si_idx), ":\n")
                                        print("new BOTTOM class classified: ", not_top_class,"\n")
                                        print("NN result: " , sampled_img_result_, "\n")
                                        print("Classes ranking: ", sampled_ranking, "\n\n")
                                    else
                                        print("new BOTTOM class classified: ", not_top_class,"\n\n")
                                    end
                                    time = now() - start_time
                                    classes_classification_time[not_top_class] = Float64(time.value)
                                    classes_classification_part[not_top_class] = "random sampling"
                                    random_sampling_classified += 1
                                    stop_sampling_timer = now()
                                end
                            end
                        end


                        if Float64((now() - stop_sampling_timer).value)/60000 >= random_sampling_duration
                            keep_generating = false
                            labels_rankings_avg = labels_rankings_avg ./ si_idx
                        end

                        si_idx += 1

                    end


                    print("labels rankings average: ", labels_rankings_avg, "\n")
                    average_ranking = sortperm(labels_rankings_avg)
                    print("labels average ranking: ", average_ranking, "\n\n")

                    random_sampling_time = now() -start_time

                    print("**************** PRUNING RELATIONS USING EXTREME SAMPLED IMAGES ****************\n\n")

                    corner_sampling_start_time = now()
                    si_idx = 1
                    stop_sampling_timer = now()
                    while true

                        if all(x -> x != -1.0, classes_classification)  # all labels classified
                            break
                        end

                        sampled_img = sample_img(img_org, eps, true)
                        sampled_img = convert(Array{Float64}, sampled_img)
                        if dataset == "pascal-voc"
                            padded_shape_ = (1, size(sampled_img)...)
                        else
                            padded_shape_ = (1, size(sampled_img)..., 1)
                        end
                        sampled_img = reshape(sampled_img, padded_shape_)
                        sampled_img_result_ = ToSoftmax(model(sampled_img))
                        sampled_ranking = reverse(sortperm(sampled_img_result_))

                        # update matrixes according to image scores
                        (gtm, ltm, gtm_diff, ltm_diff) = updateMatrixes(sampled_img_result_, gtm, ltm)

                        if gtm_diff + ltm_diff > 0
                            overall_updated += (gtm_diff + ltm_diff)
                            print("EXTREME sampled image ", string(si_idx), ":\n")
                            print("# updated entries = ", gtm_diff + ltm_diff, "\n")
                            print("# overall updated entries = ", overall_updated, "\n")
                            print("NN result: " , sampled_img_result_, "\n")
                            print("Classes ranking: ", sampled_ranking, "\n\n")
                        end

                        sampled_topk = sampled_ranking[1:k]
                        for top_class in topk
                            if !(top_class in sampled_topk)
                                prev = classes_classification[top_class]
                                classes_classification[top_class] = 2
                                if prev != 2
                                    if gtm_diff + ltm_diff == 0
                                        print("sampled image ", string(si_idx), ":\n")
                                        print("new TOPK class classified: ", top_class,"\n")
                                        print("NN result: " , sampled_img_result_, "\n")
                                        print("Classes ranking: ", sampled_ranking, "\n\n")
                                    else
                                        print("new TOPK class classified: ", top_class,"\n\n")
                                    end
                                    time = now() - start_time
                                    classes_classification_time[top_class] = Float64(time.value)
                                    classes_classification_part[top_class] = "corner sampling"
                                    corner_sampling_classified += 1
                                end
                            end
                        end
                        for not_top_class in not_topk
                            if not_top_class in sampled_topk
                                prev = classes_classification[not_top_class]
                                classes_classification[not_top_class] = 2
                                if prev != 2
                                    if gtm_diff + ltm_diff == 0
                                        print("sampled image ", string(si_idx), ":\n")
                                        print("new BOTTOM class classified: ", not_top_class,"\n")
                                        print("NN result: " , sampled_img_result_, "\n")
                                        print("Classes ranking: ", sampled_ranking, "\n\n")
                                    else
                                        print("new BOTTOM class classified: ", not_top_class,"\n\n")
                                    end
                                    time = now() - start_time
                                    classes_classification_time[not_top_class] = Float64(time.value)
                                    classes_classification_part[not_top_class] = "corner sampling"
                                    corner_sampling_classified += 1
                                end
                            end
                        end

                        if Float64((now() - stop_sampling_timer).value)/60000 >= corners_sampling_duration
                            break
                        end

                        si_idx += 1

                    end

                    print("Classes classification = ", classes_classification, "\n\n")
                else
                    print("skipping sampling...\n\n")
                end

                corner_sampling_time = now() - corner_sampling_start_time


                print("**************** PRUNING RELATIONS USING SWAP ANALYSIS ****************\n\n")

                swaps_start_time = now()
                attacks_classified = 0
                swap_milps_classified = 0
                constraints_buffer = []

                always_better_count = zeros(m)

                if skip_attacking==false

                    successful_swaps = zeros(m, m)      # -1 = failed swap attack, 1 = successful swap attack

                    top_average_ranking = filter!(top_class -> top_class in topk, copy(average_ranking))
                    bottom_average_ranking = filter!(bottom_class -> bottom_class in not_topk, copy(average_ranking))
                    
                    swap_files = Dict()
                    for top_class in topk
                        if classes_classification[top_class] == -1
                            print("Attacking the TOPK label ", string(top_class), ":\n\n")
                            for bottom_class in bottom_average_ranking
                                print("Attempting a SWAP ATTACK between labels ", string(top_class), " and ", string(bottom_class), ".\n")
                                target_labels = filter!(x -> x!=top_class, push!(copy(topk), bottom_class))   #remove the top class from the topk and add the bottom class instaed 
                                target_labels = target_labels .- 1      # the -1 is due the difference in indexing in python and julia
                                target_labels_str = replace(string(target_labels), " " => "")
                                eps_ = eps
                                command = "python -u plot_main_attack.py --k_value $k --eps $eps_ --app target_attack --target_labels $target_labels_str --label_difficult customized --data $(dataset)_samples --dataset $(uppercase(dataset)) --results $dataset --num_classes $m --arch cnn --defense $(split(model_name, '_')[2]) --complex $complex --image_size $(size(img_org)[1]) --remove_tier_para 0 --norm lInf --sample_name $(img_files[i_idx])"
                                attack_result_path = joinpath("plot_result_singles", "$(uppercase(dataset))", "customized", "target_attack", "eps_$eps_", "lInf_norm", "def_$(split(model_name, '_')[2])", "image_result_k_$(k)_sample_$(i)_$target_labels_str.npy")
                                attack_failed_path = joinpath("TKML-AP-master", "attack_failed_$(top_class)_$(bottom_class).txt")
                                perturbed_np_path = joinpath("TKML-AP-master", "perturbed_attack_$(top_class)_$(bottom_class).npy")
                                
                                if isfile(joinpath("TKML-AP-master", attack_result_path))
                                    rm(joinpath("TKML-AP-master", attack_result_path))
                                end
                                if isfile(perturbed_np_path)
                                    rm(perturbed_np_path)
                                end
                                if isfile(attack_failed_path)
                                    rm(attack_failed_path)
                                end

                                file = open(joinpath("TKML-AP-master", "pyCommand_attack_$(top_class)_$(bottom_class).txt"), "w")
                                write(file, command*"\n"*attack_result_path*"\n"*"attack_failed_$(top_class)_$(bottom_class).txt"*"\n"*"perturbed_attack_$(top_class)_$(bottom_class).npy")
                                close(file)

                                swap_files[(top_class, bottom_class)] = (attack_failed_path, perturbed_np_path)

                            end
                        end
                    end

                    file = open(joinpath("TKML-AP-master", "tmp_pyCommand_attacks_ready.txt"), "w")
                    close(file)
                    mv(joinpath("TKML-AP-master", "tmp_pyCommand_attacks_ready.txt"), joinpath("TKML-AP-master", "pyCommand_attacks_ready.txt"))


                    while isempty(swap_files) == false

                        for swap_key in keys(swap_files)

                            attack_failed_path = swap_files[swap_key][1]
                            perturbed_np_path = swap_files[swap_key][2]
                            (top_class, bottom_class) = swap_key

                            if isfile(attack_failed_path)

                                print("attack FAILED.\n\n")
                                rm(attack_failed_path)
                                successful_swaps[top_class, bottom_class] = -1

                                if !skip_mini_milps

                                    mini_milps_num += 1

                                    d = MIPVerifyMulti.relation_feasibility(
                                        model, 
                                        img_, 
                                        Gurobi.Optimizer, 
                                        Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                            "MIPGap" => 0.9999),  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                                        epsilons=[eps],
                                        l1=top_class,
                                        l2=bottom_class,
                                        img_idx = i,
                                        limit_time = mini_milp_time_limit,
                                        d = d;
                                    )

                                    feasible = !(d[:SolveStatus] == MOI.INFEASIBLE || d[:SolveStatus] == MOI.INFEASIBLE_OR_UNBOUNDED || d[:SolveStatus] == MOI.DUAL_INFEASIBLE)
                                    print(feasible ? "Couldn't prove infeasibility...\n\n" : "INFEASIBLE ==> label $top_class > label $bottom_class always\n\n")

                                    if !feasible
                                        always_better_count[top_class] += 1
                                        always_better_count[bottom_class] += 1
                                        constraint = Dict()
                                        constraint["l1"] = top_class
                                        constraint["relation"] = ">"
                                        constraint["l2"] = bottom_class
                                        push!(constraints_buffer, constraint)
                                    end

                                end

                                delete!(swap_files, swap_key)

                            end

                            if isfile(perturbed_np_path)

                                perturbed_img = np.load(perturbed_np_path)
                                if dataset=="pascal-voc"
                                    perturbed_img = imresize(perturbed_img, 300, 300)
                                    perturbed_img = float.(channelview(perturbed_img))
                                    perturbed_img = permutedims(perturbed_img, (2, 3, 1))
                                    padded_shape_ = (1, size(perturbed_img)...)
                                    perturbed_img = reshape(perturbed_img, padded_shape_)
                                else
                                    perturbed_img = convert(Array{Float64}, perturbed_img)
                                    padded_shape_ = (1, size(perturbed_img)..., 1)
                                    perturbed_img = reshape(perturbed_img, padded_shape_)
                                end
                                perturbed_img_result_ = ToSoftmax(model(perturbed_img))
                                perturbed_ranking = reverse(sortperm(perturbed_img_result_))

                                # update matrixes according to image scores
                                (gtm, ltm, gtm_diff, ltm_diff) = updateMatrixes(perturbed_img_result_, gtm, ltm)

                                overall_updated += (gtm_diff + ltm_diff)
                                print("attack SUCCEEDED!!!\n\n")
                                print("# updated entries = ", gtm_diff + ltm_diff, "\n")
                                print("# overall updated entries = ", overall_updated, "\n")
                                print("NN result: " , perturbed_img_result_, "\n")
                                print("Classes ranking: ", perturbed_ranking, "\n\n")
                    
                                perturbed_topk = perturbed_ranking[1:k]
                                for top_class in topk
                                    if !(top_class in perturbed_topk)
                                        prev = classes_classification[top_class]
                                        classes_classification[top_class] = 2
                                        if prev!=2
                                            print("new TOPK class classified: ", top_class,"\n\n")
                                            time = now() - start_time
                                            classes_classification_time[top_class] = Float64(time.value)
                                            classes_classification_part[top_class] = "attacks"
                                            attacks_classified += 1
                                        end
                                    end
                                end
                                for not_top_class in not_topk
                                    if not_top_class in perturbed_topk
                                        prev = classes_classification[not_top_class]
                                        classes_classification[not_top_class] = 2
                                        if prev!=2
                                            print("new BOTTOM class classified: ", not_top_class,"\n\n")
                                            time = now() - start_time
                                            classes_classification_time[not_top_class] = Float64(time.value)
                                            classes_classification_part[not_top_class] = "attacks"
                                            attacks_classified += 1
                                        end
                                    end
                                end

                                rm(perturbed_np_path)

                                successful_swaps[top_class, bottom_class] = 1

                                delete!(swap_files, swap_key)

                            end

                        end

                        # found one successful swap (topk class can also be a bottom class)
                        # if successful_swaps[top_class, bottom_class] == 1
                        #     break
                        # end

                    end

                    # top class always has a higher score than (m-k) other classes
                    if always_better_count[top_class] == m - k
                        classes_classification[top_class] = 0
                        print("new TOP class classified: ", top_class,"\n\n")
                        time = now() - start_time
                        classes_classification_time[top_class] = Float64(time.value)
                        classes_classification_part[top_class] = "swap-milps"
                        swap_milps_classified += 1
                    end

                    
                    for bottom_class in not_topk
                        if classes_classification[bottom_class] == -1
                            print("Attacking the BOTTOM label ", string(bottom_class), ":\n\n")
                            for top_class in reverse(top_average_ranking)
                                print("Attempting a SWAP ATTACK between labels ", string(top_class), " and ", string(bottom_class), ".\n")
                                if successful_swaps[top_class, bottom_class] == -1   # already tried this swap in topk attacks
                                    print("attack FAILED (from previous attempt).\n\n")
                                    continue
                                end
                                target_labels = filter!(x -> x!=top_class, push!(copy(topk), bottom_class))   #remove the top class from the topk and add the bottom class instaed 
                                target_labels = target_labels .- 1      # the -1 is due the difference in indexing in python and julia
                                target_labels_str = replace(string(target_labels), " " => "")
                                eps_ = eps
                                command = "python -u plot_main_attack.py --k_value $k --eps $eps_ --app target_attack --target_labels $target_labels_str --label_difficult customized --data $(dataset)_samples --dataset $(uppercase(dataset)) --results $dataset --num_classes $m --arch cnn --defense $(split(model_name, '_')[2]) --complex $complex --image_size $(size(img_org)[1]) --remove_tier_para 0 --norm lInf --sample_name $(img_files[i_idx])"
                                attack_result_path = joinpath("plot_result_singles", "$(uppercase(dataset))", "customized", "target_attack", "eps_$eps_", "lInf_norm", "def_$(split(model_name, '_')[2])", "image_result_k_$(k)_sample_$(i)_$target_labels_str.npy")
                                attack_failed_path = joinpath("TKML-AP-master", "attack_failed_$(top_class)_$(bottom_class).txt")
                                perturbed_np_path = joinpath("TKML-AP-master", "perturbed_attack_$(top_class)_$(bottom_class).npy")
                                
                                if isfile(joinpath("TKML-AP-master", attack_result_path))
                                    rm(joinpath("TKML-AP-master", attack_result_path))
                                end
                                if isfile(perturbed_np_path)
                                    rm(perturbed_np_path)
                                end
                                if isfile(attack_failed_path)
                                    rm(attack_failed_path)
                                end

                                file = open(joinpath("TKML-AP-master", "pyCommand_attack_$(top_class)_$(bottom_class).txt"), "w")
                                write(file, command*"\n"*attack_result_path*"\n"*"attack_failed_$(top_class)_$(bottom_class).txt"*"\n"*"perturbed_attack_$(top_class)_$(bottom_class).npy")
                                close(file)

                                swap_files[(top_class, bottom_class)] = (attack_failed_path, perturbed_np_path)

                            end
                        end
                    end

                    file = open(joinpath("TKML-AP-master", "tmp_pyCommand_attacks_ready.txt"), "w")
                    close(file)
                    mv(joinpath("TKML-AP-master", "tmp_pyCommand_attacks_ready.txt"), joinpath("TKML-AP-master", "pyCommand_attacks_ready.txt"))
                                
                    while isempty(swap_files) == false

                        for swap_key in keys(swap_files)

                            attack_failed_path = swap_files[swap_key][1]
                            perturbed_np_path = swap_files[swap_key][2]
                            (top_class, bottom_class) = swap_key

                            if isfile(attack_failed_path)
                                print("attack FAILED.\n\n")
                                rm(attack_failed_path)
                                successful_swaps[top_class, bottom_class] = -1

                                if !skip_mini_milps

                                    mini_milps_num += 1

                                    d = MIPVerifyMulti.relation_feasibility(
                                        model, 
                                        img_, 
                                        Gurobi.Optimizer, 
                                        Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                            "MIPGap" => 0.9999),  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                                        epsilons=[eps],
                                        l1=top_class,
                                        l2=bottom_class,
                                        img_idx = i,
                                        limit_time = mini_milp_time_limit,
                                        d = d;
                                    )

                                    feasible = !(d[:SolveStatus] == MOI.INFEASIBLE || d[:SolveStatus] == MOI.INFEASIBLE_OR_UNBOUNDED || d[:SolveStatus] == MOI.DUAL_INFEASIBLE)
                                    print(feasible ? "Couldn't prove infeasibility...\n\n" : "INFEASIBLE ==> label $top_class > label $bottom_class always\n\n")

                                    if !feasible
                                        always_better_count[top_class] += 1
                                        always_better_count[bottom_class] += 1
                                        constraint = Dict()
                                        constraint["l1"] = top_class
                                        constraint["relation"] = ">"
                                        constraint["l2"] = bottom_class
                                        push!(constraints_buffer, constraint)
                                    end

                                end

                                delete!(swap_files, swap_key)

                            end

                            if isfile(perturbed_np_path)
                                perturbed_img = np.load(perturbed_np_path)
                                if dataset=="pascal-voc"
                                    perturbed_img = imresize(perturbed_img, 300, 300)
                                    perturbed_img = float.(channelview(perturbed_img))
                                    perturbed_img = permutedims(perturbed_img, (2, 3, 1))
                                    padded_shape_ = (1, size(perturbed_img)...)
                                    perturbed_img = reshape(perturbed_img, padded_shape_)
                                else
                                    perturbed_img = convert(Array{Float64}, perturbed_img)
                                    padded_shape_ = (1, size(perturbed_img)..., 1)
                                    perturbed_img = reshape(perturbed_img, padded_shape_)
                                end
                                perturbed_img_result_ = ToSoftmax(model(perturbed_img))
                                perturbed_ranking = reverse(sortperm(perturbed_img_result_))

                                # update matrixes according to image scores
                                (gtm, ltm, gtm_diff, ltm_diff) = updateMatrixes(perturbed_img_result_, gtm, ltm)

                                overall_updated += (gtm_diff + ltm_diff)
                                print("attack SUCCEEDED!!!\n\n")
                                print("# updated entries = ", gtm_diff + ltm_diff, "\n")
                                print("# overall updated entries = ", overall_updated, "\n")
                                print("NN result: " , perturbed_img_result_, "\n")
                                print("Classes ranking: ", perturbed_ranking, "\n\n")
                    
                                perturbed_topk = perturbed_ranking[1:k]
                                for top_class in topk
                                    if !(top_class in perturbed_topk)
                                        prev = classes_classification[top_class]
                                        classes_classification[top_class] = 2
                                        if prev!=2
                                            print("new TOPK class classified: ", top_class,"\n\n")
                                            time = now() - start_time
                                            classes_classification_time[top_class] = Float64(time.value)
                                            classes_classification_part[top_class] = "attacks"
                                            attacks_classified += 1
                                        end
                                    end
                                end
                                for not_top_class in not_topk
                                    if not_top_class in perturbed_topk
                                        prev = classes_classification[not_top_class]
                                        classes_classification[not_top_class] = 2
                                        if prev!=2
                                            print("new BOTTOM class classified: ", not_top_class,"\n\n")
                                            time = now() - start_time
                                            classes_classification_time[not_top_class] = Float64(time.value)
                                            classes_classification_part[not_top_class] = "attacks"
                                            attacks_classified += 1
                                        end
                                    end
                                end

                                rm(perturbed_np_path)

                                successful_swaps[top_class, bottom_class] = 1

                                delete!(swap_files, swap_key)

                            end

                        end

                        # # found one successful swap (topk class can also be a bottom class)
                        # if successful_swaps[top_class, bottom_class] == 1
                        #     break
                        # end

                    end

                    # bottom class always has a lower score than k other classes
                    if always_better_count[bottom_class] == k
                        classes_classification[bottom_class] = 1
                        print("new BOTTOM class classified: ", bottom_class,"\n\n")
                        time = now() - start_time
                        classes_classification_time[bottom_class] = Float64(time.value)
                        classes_classification_part[bottom_class] = "swap-milps"
                        swap_milps_classified += 1
                    end

                    print("Classes classification = ", classes_classification, "\n\n")

                else
                    print("skipping attacks...\n\n")
                end

                swaps_time = now() - swaps_start_time


                print("**************** CLASSIFYING LABELS USING A MILP VERIFIER ****************\n\n")

                verifier_start_time = now()
                verifier_classified = 0

                constraints_buffer_sub = copy(constraints_buffer)

                for (top_idx, top_class) in enumerate(topk)
                    if all(x -> x == 1, classes_classification[not_topk]) && classes_classification[top_class] == -1
                        classes_classification[top_class] = 0   # always in topk
                        print("new TOP class classified: ", top_class,"\n\n")
                        time = now() - start_time
                        classes_classification_time[top_class] = Float64(time.value)
                        classes_classification_part[top_class] = "full L-out"
                        # verifier_classified += 1
                        continue
                    end
                    if classes_classification[top_class] == -1
                        if !skip_sub_milps
                            print("Checking robustness for top label:\n")
                            print("SUB-MILP: Does the label \"", string(top_class), "\" stays IN the top-$k labels in the sub-neighborhood [x+$(eps-mini_eps), x+$eps]?\n\n")

                            sub_d = MIPVerifyMulti.find_adversarial_example(
                                model, 
                                img_, 
                                top_class, 
                                invert_target_selection = true,
                                Gurobi.Optimizer, 
                                Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                        "MIPGap" => 0.9999),  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                                epsilons=[eps],
                                norm_order = Inf,
                                multilabel = k,
                                img_idx = i,
                                optimal = "max",
                                swap_consts_buf = constraints_buffer_sub,
                                d = sub_d;
                            )

                            sub_milps_num += 1

                            constraints_buffer_sub = []

                            print("\n\nRL = "*string(RL)*"\n")
                            robust = (RL>0)
                        else
                            robust = true
                        end
                        
                        if robust
                            if !skip_sub_milps
                                print("ROBUST\n")
                                print("FAILED to refute robustness of $top_class using SUB-MILP.\n\n")
                            else
                                print("skipping sub-milps...\n\n")
                            end

                            top_C = filter(x -> classes_classification[x] == -1, topk)
                            
                            while length(top_C) > 0

                                if skip_super_milps
                                    print("skipping super-milps...\n\n")
                                    break
                                end

                                print("Checking robustness for top labels:\n")
                                print("SUPER-MILP: Do the labels $top_C stays IN the top-$k labels in the eps=", eps,"-ball?\n\n")

                                # set_optimizer_attribute(m, "BestObjStop", 0)  # for early stopping (a non robust solution (adv example) is found)
                                # set_optimizer_attribute(m, "MIPGap", 0.99)  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                                # set_optimizer_attribute(m, "Presolve", 0)

                                d = MIPVerifyMulti.find_adversarial_example(
                                    model, 
                                    img_, 
                                    top_C, 
                                    invert_target_selection = true,
                                    Gurobi.Optimizer, 
                                    Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                        "MIPGap" => 0.9999,  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                                        "MIPFocus" => 1,
                                        "Heuristics" => DEFAULT_HEURISTICS_PARAMETER,
                                        "ImproveStartTime" => 0.0),
                                    epsilons=[eps],
                                    norm_order = Inf,
                                    multilabel = k,
                                    img_idx = i,
                                    optimal = "max",
                                    swap_consts_buf = constraints_buffer,
                                    d = d;
                                )

                                super_milps_num += 1

                                feasible = !(d[:SolveStatus] == MOI.INFEASIBLE || d[:SolveStatus] == MOI.INFEASIBLE_OR_UNBOUNDED || d[:SolveStatus] == MOI.DUAL_INFEASIBLE)
                                robust = !feasible

                                constraints_buffer = []
                                # RL = -d[:BestObjective]
                                # print("\n\nRL = "*string(RL)*"\n")
                                # robust = (RL>0)

                                if robust
                                    print("ROBUST\n\n")
                                    for top_class_C in top_C
                                        classes_classification[top_class_C] = 0   # always in topk
                                        print("new TOP class classified: ", top_class_C,"\n\n")
                                        time = now() - start_time
                                        classes_classification_time[top_class_C] = Float64(time.value)
                                        classes_classification_part[top_class_C] = "MILP verifier"
                                        verifier_classified += 1
                                    end
                                    break
                                else
                                    print("NOT ROBUST\n\n")                        
                                    classes_classification[d[:OptNonTarget]] = 2   # can be in both
                                    print("new TOP class classified: ", d[:OptNonTarget],"\n\n")
                                    time = now() - start_time
                                    classes_classification_time[d[:OptNonTarget]] = Float64(time.value)
                                    classes_classification_part[d[:OptNonTarget]] = "MILP verifier"
                                    verifier_classified += 1
                                    top_C = filter(x -> x != d[:OptNonTarget], top_C)
                                end

                            end
                        
                        else
                            print("NOT ROBUST\n\n")
                            classes_classification[top_class] = 2   # can be in both
                            print("new TOP class classified: ", top_class,"\n\n")
                            time = now() - start_time
                            classes_classification_time[top_class] = Float64(time.value)
                            classes_classification_part[top_class] = "MILP verifier"
                            verifier_classified += 1
                        end
                    end
                end

                for top_class in topk
                    if classes_classification[top_class] == -1
                        print("Checking robustness for topk label:\n")
                        print("Does the label \"", string(top_class), "\" stays IN the top-$k labels in the eps=", eps,"-ball?\n\n")

                        d = MIPVerifyMulti.find_adversarial_example(
                                model, 
                                img_, 
                                top_class, 
                                invert_target_selection = true,
                                Gurobi.Optimizer, 
                                Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                        "MIPGap" => 0.9999),  # for early stopping (lower and upper bounds of optimal solution have the same sign),
                                epsilons=[eps],
                                norm_order = Inf,
                                multilabel = k,
                                img_idx = i,
                                optimal = "max",
                                swap_consts_buf = constraints_buffer,
                                d = d;
                            )

                        regular_milps_num += 1
                        
                        constraints_buffer = []
                        RL = -d[:BestObjective]
                        print("\n\nRL = "*string(RL)*"\n")
                        robust = (RL>0)
                        if robust
                            print("ROBUST\n\n")
                            classes_classification[top_class] = 0   # always in topk
                        else
                            print("NOT ROBUST\n\n")
                            classes_classification[top_class] = 2   # can be in both
                        end
                        print("new TOPK class classified: ", top_class,"\n\n")
                        time = now() - start_time
                        classes_classification_time[top_class] = Float64(time.value)
                        classes_classification_part[top_class] = "MILP verifier"
                        verifier_classified += 1
                    end
                end


                for (bottom_idx, bottom_class) in enumerate(not_topk)
                    if all(x -> x == 1, classes_classification[topk]) && classes_classification[bottom_class] == -1
                        classes_classification[bottom_class] = 1   # always in non-topk
                        print("new BOTTOM class classified: ", bottom_class,"\n\n")
                        time = now() - start_time
                        classes_classification_time[bottom_class] = Float64(time.value)
                        classes_classification_part[bottom_class] = "full L-in"
                        # verifier_classified += 1
                        continue
                    end
                    if classes_classification[bottom_class] == -1
                        if !skip_sub_milps
                            print("Checking robustness for bottom label:\n")
                            print("SUB-MILP: Does the label \"", string(bottom_class), "\" stays OUT of the top-$k labels in the sub-neighborhood [x+$(eps-mini_eps), x+$eps]?\n\n")

                            sub_d = MIPVerifyMulti.find_adversarial_example(
                                model, 
                                img_, 
                                bottom_class, 
                                invert_target_selection = true,
                                Gurobi.Optimizer, 
                                Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                        "MIPGap" => 0.9999),  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                                epsilons=[eps],
                                norm_order = Inf,
                                multilabel = k,
                                img_idx = i,
                                optimal = "min",
                                swap_consts_buf = constraints_buffer_sub,
                                d = sub_d;
                            )

                            sub_milps_num += 1

                            constraints_buffer_sub = []
                            RL = sub_d[:BestObjective]
                            print("\n\nRL = "*string(RL)*"\n")
                            robust = (RL>0)
                        else
                            robust = true
                        end
                        
                        if robust
                            if !skip_sub_milps
                                print("ROBUST\n")
                                print("FAILED to refute robustness of $bottom_class using SUB-MILP.\n\n")
                            else
                                print("skipping sub-milps...\n\n")
                            end

                            bottom_C = filter(x -> classes_classification[x] == -1, not_topk)
                            
                            while length(bottom_C) > 0

                                if skip_super_milps
                                    print("skipping super-milps...\n\n")
                                    break
                                end

                                print("Checking robustness for bottom labels:\n")
                                print("SUPER-MILP: Do the labels $bottom_C stays OUT of the top-$k labels in the eps=", eps,"-ball?\n\n")

                                d = MIPVerifyMulti.find_adversarial_example(
                                    model, 
                                    img_, 
                                    bottom_C, 
                                    invert_target_selection = true,
                                    Gurobi.Optimizer, 
                                    Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                        "MIPGap" => 0.9999,  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                                        "MIPFocus" => 1,
                                        "Heuristics" => DEFAULT_HEURISTICS_PARAMETER,
                                        "ImproveStartTime" => 0.0),
                                    epsilons=[eps],
                                    norm_order = Inf,
                                    multilabel = k,
                                    img_idx = i,
                                    optimal = "min",
                                    swap_consts_buf = constraints_buffer,
                                    d = d;
                                )

                                super_milps_num += 1

                                feasible = !(d[:SolveStatus] == MOI.INFEASIBLE || d[:SolveStatus] == MOI.INFEASIBLE_OR_UNBOUNDED || d[:SolveStatus] == MOI.DUAL_INFEASIBLE)
                                robust = !feasible

                                constraints_buffer = []
                                # RL = -d[:BestObjective]
                                # print("\n\nRL = "*string(RL)*"\n")
                                # robust = (RL>0)

                                if robust
                                    print("ROBUST\n\n")
                                    for bottom_class_C in bottom_C
                                        classes_classification[bottom_class_C] = 1   # always not in topk
                                        print("new BOTTOM class classified: ", bottom_class_C,"\n\n")
                                        time = now() - start_time
                                        classes_classification_time[bottom_class_C] = Float64(time.value)
                                        classes_classification_part[bottom_class_C] = "MILP verifier"
                                        verifier_classified += 1
                                    end
                                    break
                                else
                                    print("NOT ROBUST\n\n")
                                    classes_classification[d[:OptNonTarget]] = 2   # can be in both
                                    print("new BOTTOM class classified: ", d[:OptNonTarget],"\n\n")
                                    time = now() - start_time
                                    classes_classification_time[d[:OptNonTarget]] = Float64(time.value)
                                    classes_classification_part[d[:OptNonTarget]] = "MILP verifier"
                                    verifier_classified += 1

                                    bottom_C = filter(x -> x != d[:OptNonTarget], bottom_C)
                                end

                            end
                        
                        else
                            print("NOT ROBUST\n\n")
                            classes_classification[bottom_class] = 2   # can be in both
                            print("new BOTTOM class classified: ", bottom_class,"\n\n")
                            time = now() - start_time
                            classes_classification_time[bottom_class] = Float64(time.value)
                            classes_classification_part[bottom_class] = "MILP verifier"
                            verifier_classified += 1
                        end
                    end
                end

                for bottom_class in not_topk
                    if classes_classification[bottom_class] == -1
                        print("Checking robustness for bottom label:\n")
                        print("Regular MILP: Does the label \"", string(bottom_class), "\" stays OUT of the top-$k labels in the eps=", eps,"-ball?\n\n")

                        d = MIPVerifyMulti.find_adversarial_example(
                            model, 
                            img_, 
                            bottom_class, 
                            invert_target_selection = true,
                            Gurobi.Optimizer, 
                            Dict("BestObjStop" => 0, # for early stopping (a non robust solution (adv example) is found)
                                        "MIPGap" => 0.9999),  # for early stopping (lower and upper bounds of optimal solution have the same sign)
                            epsilons=[eps],
                            norm_order = Inf,
                            multilabel = k,
                            img_idx = i,
                            optimal = "min",
                            swap_consts_buf = constraints_buffer,
                            d = d;
                        )

                        regular_milps_num += 1

                        constraints_buffer = []
                        RL = d[:BestObjective]
                        print("\n\nRL = "*string(RL)*"\n")
                        robust = (RL>0)
                        if robust
                            print("ROBUST\n\n")
                            classes_classification[bottom_class] = 1   # always not in topk
                        else
                            print("NOT ROBUST\n\n")
                            classes_classification[bottom_class] = 2   # can be in both
                        end
                        print("new BOTTOM class classified: ", bottom_class,"\n\n")
                        time = now() - start_time
                        classes_classification_time[bottom_class] = Float64(time.value)
                        classes_classification_part[bottom_class] = "MILP verifier"
                        verifier_classified += 1
                    end
                end

                print("Final classes classification = ", classes_classification, "\n\n")
                verifier_time = now() - verifier_start_time

                
                df1 = DataFrame(Label = ranking, 
                Scores = result_[ranking],
                Classification = translate_classification(classes_classification, k)[ranking],
                Time_till_classified_minutes = classes_classification_time[ranking]./60000,
                Part_of_classification = classes_classification_part[ranking])
                CSV.write(joinpath("$output_path", "classification_summary.csv"), df1, writeheader=true)

                df2 = DataFrame(Part = ["Random Sampling", "Corner Sampling", "Attacks", "Swap-Milps", "Total swap analysis", "MILP Verifier", "Total"],
                Execution_time_minutes = [Float64(random_sampling_time.value), Float64(corner_sampling_time.value), "-", "-", Float64(swaps_time.value), Float64(verifier_time.value), Float64(random_sampling_time.value)+Float64(corner_sampling_time.value)+Float64(attacks_time.value)+Float64(swap_milps_time.value)+Float64(verifier_time.value)]./60000,
                Num_of_labels_classified = [random_sampling_classified, corner_sampling_classified, attacks_classified, swap_milps_classified, (attacks_classified + swap_milps_classified), verifier_classified, random_sampling_classified + corner_sampling_classified + attacks_classified + swap_milps_classified + verifier_classified])
                CSV.write(joinpath("$output_path", "exec_summary.csv"), df2, writeheader=true)

                df3 = DataFrame(MILP_TYPE = ["SWAP-MILP", "REGULAR-MILP", "SUPER-MILP"],
                TIMES = [mini_milps_num, regular_milps_num, super_milps_num])
                CSV.write(joinpath("$output_path", "milp_summary.csv"), df3, writeheader=true)

                df4 = DataFrame(PARAM = ["dataset", "model", "input", "num_of_classes", "K", "epsilon", "skip_sampling", "skip_attacking", "skip_mini_milps", "skip_super_milps", "random_sampling_duration[mins]", "corners_sampling_duration[mins]", "mini_milp_time_limit[secs]"], 
                VALUE = [dataset, model_name, i, m, k, eps, skip_sampling, skip_attacking, skip_mini_milps, skip_super_milps, random_sampling_duration, corners_sampling_duration, mini_milp_time_limit])
                CSV.write(joinpath("$output_path", "params_summary.csv"), df4, writeheader=true)

                if till_img_idx == i_idx
                    break
                end

            end

        end
    end

end