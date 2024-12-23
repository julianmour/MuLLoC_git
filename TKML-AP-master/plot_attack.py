# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:12:49 2021
@author: Lipeng Ke
"""

from tqdm import tqdm
import torch
import gc
import os
from utils import get_ap_score, generate_target_zeros, generate_target_zeros_3_cases
import numpy as np
import torch
import numpy as np
from evaluate_metrics import hamming_loss, FR, TP_index, nontargeted_TP_index, topk_acc_metric, topk_acc_metric_1_to_10
from models.projection import clip_eta
from torch.autograd import Variable
import time
import ast


def l2_topk_non_targeted_attack(model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, device,
                                Projection_flag=False, lr=1e-2):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    modifier = Variable(torch.zeros(*shape).cuda(), requires_grad=True).to(device)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr)

    purtabed_img = torch.zeros(*shape)
    purtabed_final_out = np.zeros(tlab.shape)
    predict_label_final = np.zeros(tlab.shape)
    converge_iter = maxiter
    attack_success = False
    c_list = [0] * 6
    for iteration in range(maxiter):
        optimizer.zero_grad()
        purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)

        if Projection_flag:
            purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta

        purtabed_out = model(purtabed_img)

        # loss
        real = torch.max(tlab * purtabed_out - ((1 - tlab) * 10000))

        # t_value = torch.max(torch.zeros(purtabed_out.shape).to(device), real - purtabed_out)
        t_value = real - purtabed_out
        lambda_l, _ = torch.topk(t_value, label.shape[1] - k_value)
        # loss = (label.shape[1]-k_value)*lambda_l[:,-1] + torch.sum(torch.max(torch.zeros(t_value.shape).to(device), t_value - lambda_l[:,-1]))
        loss = lambda_l[:, -1] + (1 / (label.shape[1] - k_value)) * torch.sum(
            torch.max(torch.zeros(t_value.shape).to(device), t_value - lambda_l[:, -1]))
        loss = torch.sum(loss)

        # Calculate gradient
        loss.backward()
        optimizer.step()

        Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(),
                                                   k_value)
        purtabed_final_out = purtabed_out.cpu().detach().numpy()
        predict_label_final = predict_label

        # If attack success terminate and return
        if Flag:
            converge_iter = iteration
            attack_success = True
            purtabed_final_out = purtabed_out.cpu().detach().numpy()
            predict_label_final = predict_label
            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
                topk_acc_metric_1_to_10(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
            print('iter:', iteration, 'loss= ', "{}".format(loss), \
                  'attacked: ', Flag, 'predict_label:', predict_label, \
                  'GT:', label.cpu().detach().numpy(), \
                  'min:', "{:.5f}".format(modifier.min().cpu()), \
                  'max:', "{:.5f}".format(modifier.max().cpu()), \
                  'norm:', "{:.5f}".format(np.linalg.norm(eta.cpu().detach().numpy())))
            break
            # return purtabed_img, modifier, iteration
    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    return purtabed_img_out, modifier_out, converge_iter, attack_success, c_list, purtabed_final_out, predict_label_final


def l2_topk_targeted_attack(model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, targets_zeros, GT, device,
                            remove_tier_para=0.1, lr=1e-2, norm='l2'):
    # torch.manual_seed(1)
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    # modifier = torch.zeros(inputs.shape, dtype=inputs.type)
    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    # variables we are going to optimize over
    if device.type == 'cuda':
        modifier = Variable(torch.zeros(*shape).cuda(), requires_grad=True).to(device)
    else:
        modifier = Variable(torch.zeros(*shape), requires_grad=True).to(device)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr)

    best_norm = 1e10
    purtabed_img_out = torch.zeros(*shape)
    modifier_out = torch.zeros(*shape)
    purtabed_final_out = np.zeros(tlab.shape)
    predict_label_final = np.zeros(tlab.shape)
    attack_success = False
    c_list = [0] * 6
    for iteration in range(maxiter):
        optimizer.zero_grad()

        # purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        # eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        # eta = clip_eta(eta, 2, const)
        purtabed_img = modifier + timg
        eta = purtabed_img - timg
        eta = clip_eta(eta, 2, const)

        # purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta
        purtabed_img = timg + eta

        purtabed_out = model(purtabed_img)

        # loss function
        lamb, _ = torch.topk(purtabed_out, k_value)
        loss0 = torch.clamp(tlab * (lamb[:, -1] - purtabed_out), min=0.0)
        loss0 = torch.sum(loss0)

        # solve tie
        loss_tie = (remove_tier_para / 2) * torch.sum(torch.square(1 - tlab * purtabed_out), [1])
        loss1 = torch.sum(loss_tie)
        loss = loss0 + loss1

        # Calculate gradient
        loss.backward()
        optimizer.step()

        TP_flag, predict_label = TP_index(targets_zeros, purtabed_out.cpu().detach().numpy())

        if norm == 'l2':
            eta_norm = np.linalg.norm(eta.cpu().detach().numpy())  # L2 attack
        elif norm == 'lInf':
            eta_norm = max(np.abs(np.ndarray.flatten(eta.cpu().detach().numpy())))  # LInf attack

        if TP_flag == True and eta_norm < best_norm:
            best_norm = eta_norm
            # purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
            purtabed_img_out = purtabed_img.cpu().detach().numpy()
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            purtabed_final_out = purtabed_out.cpu().detach().numpy()
            predict_label_final = predict_label
            attack_success = True

            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())

            print('iter:', iteration, 'loss= ', "{:.5f}".format(loss), \
                  'loss0= ', "{:.5f}".format(loss0), \
                  'loss1= ', "{:.5f}".format(loss1), \
                  'attacked: ', TP_flag, \
                  'targets: ', targets_zeros, \
                  'predict_label:', predict_label, \
                  'attack_label_scores:', purtabed_out, \
                  'GT:', GT.cpu().detach().numpy(), \
                  'min:', "{:.5f}".format(modifier.min().cpu()), \
                  'max:', "{:.5f}".format(modifier.max().cpu()), \
                  'norm:', "{:.5f}".format(eta_norm))

    return purtabed_img_out, modifier_out, attack_success, c_list, purtabed_final_out, predict_label_final


def UAP(args, model, device, val_loader, sample_list, train_index_end):
    # global_delta = torch.zeros((1, 3, 300, 300)).to(device)
    global_delta = torch.zeros(iter(val_loader).next()[0].shape).to(device)
    fooling_rate = 0
    itr = 0
    boxmul = (args.boxmax - args.boxmin) / 2.
    boxplus = (args.boxmin + args.boxmax) / 2.
    while fooling_rate < args.ufr_lower_bound and itr < args.max_iter_uni:
        print('Starting pass number ', itr)
        index1 = 0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:train_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                print('ith:', index1 + 1)
                GT = GT.int()
                print('\n')
                data, GT = data.to(device), GT.to(device)
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                # Flag, predict_label = nontargeted_TP_index(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value)
                Flag = bool(topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value))
                if Flag == False:
                    new_img, modifiered, converge_iter, attack_success, c_list = \
                        l2_topk_non_targeted_attack(model, data + global_delta, GT, args.k_value, args.eps, \
                                                    args.maxiter, args.boxmax, args.boxmin, device,
                                                    Projection_flag=False, lr=args.lr_attack)
                    if attack_success == True:
                        global_delta = global_delta + torch.from_numpy(modifiered).to(device)
                        print('pre_global_delta:', np.linalg.norm(global_delta.cpu().detach().numpy()))
                        global_delta = clip_eta(global_delta, args.uap_norm, args.uap_eps)
                        print('after_global_delta:', np.linalg.norm(global_delta.cpu().detach().numpy()))
                else:
                    print('attack had successed')

                index1 = index1 + 1
            if index1 == train_index_end:
                break

        itr = itr + 1

        count = 0
        data_num = 0
        index2 = 0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:train_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                data, GT = data.to(device), GT.to(device)
                data_num = data_num + 1
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                count = count + topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value)
                index2 = index2 + 1
            if index2 == train_index_end:
                break

        fooling_rate = count / data_num
        print('FOOLING RATE = ', fooling_rate)
    return global_delta


def save_result(success_c_list, index, args, success_img_index, success_modifier_norm_list,
                success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3, success_modifier_norm_index_4,
                success_modifier_norm_index_5):
    final_all = []
    c_list_sum = np.sum(np.asarray(success_c_list), 0) / index
    if success_c_list != []:
        print('attack_type= ', "{}".format(args.app), 'label_difficult= ', "{}".format(args.label_difficult))
        print('FR_1= ', "{:.5f}".format(c_list_sum[0]), \
              'FR_2= ', "{:.5f}".format(c_list_sum[1]), \
              'FR_3= ', "{:.5f}".format(c_list_sum[2]), \
              'FR_4= ', "{:.5f}".format(c_list_sum[3]), \
              'FR_5= ', "{:.5f}".format(c_list_sum[4]), \
              'FR_10= ', "{:.5f}".format(c_list_sum[5]))
        print('avg_norm_1= ', "{}".format(
            np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))), \
              'avg_norm_2= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))), \
              'avg_norm_3= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))), \
              'avg_norm_4= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))), \
              'avg_norm_5= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))), \
              'avg_norm_10= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))))
    else:
        print('All False')

    # final_all.append(success_perturbated_img_list)
    # final_all.append(success_modifier_list)
    # final_all.append(success_modifier_norm_list)
    final_all.append(success_img_index)
    final_all.append(success_c_list)
    final_all.append(success_modifier_norm_index_0)
    final_all.append(success_modifier_norm_index_1)
    final_all.append(success_modifier_norm_index_2)
    final_all.append(success_modifier_norm_index_3)
    final_all.append(success_modifier_norm_index_4)
    final_all.append(success_modifier_norm_index_5)
    os.makedirs('./result/{}/{}/eps_{}'.format(args.label_difficult, args.app, args.eps), exist_ok=True)
    final_all = np.asarray(final_all, dtype=object)
    np.save('./result/{}/{}/eps_{}/k_{}'.format(args.label_difficult, args.app, args.eps, args.k_value), final_all)


def tkmlap(args, model, device, val_loader):
    """
    Evaluate a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        val_dataloader: validation images dataloader

    """
    print('kvalue: ', args.k_value, 'label_difficult', args.label_difficult, 'app_type:', args.app)
    if args.dataset == "DMNIST" or args.dataset == "TMNIST" or args.dataset == "PASCAL-VOC":
        sample_list = np.arange(0, 50)
    else:
        sample_list = np.load('ap_{}_list.npy'.format(args.dataset))
    success_count = 0
    index = 0
    index_success = 0
    success_perturbated_img_list = []
    success_modifier_list = []
    success_modifier_norm_list = []
    success_img_index = []
    success_c_list = []
    success_modifier_norm_index_0 = []
    success_modifier_norm_index_1 = []
    success_modifier_norm_index_2 = []
    success_modifier_norm_index_3 = []
    success_modifier_norm_index_4 = []
    success_modifier_norm_index_5 = []
    a = {}
    if args.app == 'none_target_attack':
        # a['best_3'] = [30] #874, 49,5
        a['best_3'] = [5]  # 874, 49,5
        a['best_5'] = [5]
        a['best_10'] = [5]
    elif args.app == 'target_attack':
        if args.eps == 10:
            a['best_3'] = [1118, 488, 460]
            a['best_5'] = [310, 316, 814]
            a['best_10'] = [858, 896, 316]
            a['random_3'] = [309, 860, 828]
            a['random_5'] = [721, 390, 603]
            a['random_10'] = [1067, 978, 896]
            a['worst_3'] = [858, 859, 721]
            a['worst_5'] = [942, 870, 978]
            a['worst_10'] = [1137, 521, 858]
        elif args.eps == 2:
            a['best_3'] = [277]  # [277,313,412]
            a['best_5'] = [277]  # [277,313,412]
            a['best_10'] = [277]  # [277,313,412]
            a['random_3'] = [247]  # [247,9,  194]
            a['random_5'] = [247]  # [247, 874 ,746]
            a['random_10'] = [247]  # [247, 386, 720]
            a['worst_3'] = [647]  # [614, 647, 1177]
            a['worst_5'] = [647]  # [614, 647, 1177]
            a['worst_10'] = [647]  # [614, 647, 1177]
    elif args.app == 'UAP_attack':
        # a['best_3'] = [6566, 6568, 6569, 6570, 6571, 6573, 6574, 6575, 6580, 6584, \
        #                7138, 7145, 7161, 7167, 7168, 7169, 7175, 7177, 7186, 7318,\
        #                8101, 8103, 8114, 8127, 8153, 8161, 8168, 8171, 8178, 8184]  # 874, 49,5, 6472,6474,6475,6477,6479,6480,6481, 6482, 6483,6484
        a['best_3'] = [6574, 7138, 8114, 8184]
    model.eval()

    if args.app == 'UAP_attack':
        ###Train
        # global_delta = UAP(args, model, device, val_loader,sample_list, args.uap_train_index_end)
        ###Test
        boxmul = (args.boxmax - args.boxmin) / 2.
        boxplus = (args.boxmin + args.boxmax) / 2.
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in a['{}_{}'.format(args.label_difficult, args.k_value)]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                # if ith in sample_list[:args.uap_train_index_end]:
                print('ith_0:', ith)
                print('ith2:', index + 1)
                data, GT = data.to(device), GT.to(device)
                plot_list = []
                plot_list.append(data.cpu().detach().numpy())
                plot_list.append(GT.cpu().detach().numpy())
                plot_list.append(np.argsort(-(model(data).cpu().detach().numpy())[0]))

                global_delta = torch.from_numpy(np.load(
                    './result/{}/{}/{}/eps_{}/perturbation_{}.npy'.format(args.dataset, args.label_difficult, args.app,
                                                                          args.eps, args.k_value))).to(device)
                plot_list.append((data + global_delta).cpu().detach().numpy())
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                Flag, predict_label = nontargeted_TP_index(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value)
                plot_list.append(predict_label)
                os.makedirs(
                    './plot_result/{}/{}/{}/eps_{}'.format(args.dataset, args.label_difficult, args.app, args.eps),
                    exist_ok=True)
                plot_list = np.asarray(plot_list, dtype=object)
                np.save(
                    './plot_result/{}/{}/{}/eps_{}/images_result_k_{}_ith_{}'.format(args.dataset,
                                                                                     args.label_difficult, args.app,
                                                                                     args.eps,
                                                                                     args.k_value, ith), plot_list)
                # Flag = bool(topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value))
                if Flag == True:
                    success_count = success_count + 1
                    c_list = [0] * 6
                    c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
                        topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), pred.cpu().detach().numpy())
                    success_modifier_norm_list.append(
                        np.linalg.norm(global_delta.cpu().detach().numpy()) / ((args.image_size) * (args.image_size)))
                    success_img_index.append(ith)
                    success_c_list.append(c_list)
                    if c_list[0] == 1:
                        success_modifier_norm_index_0.append(index_success)
                    if c_list[1] == 1:
                        success_modifier_norm_index_1.append(index_success)
                    if c_list[2] == 1:
                        success_modifier_norm_index_2.append(index_success)
                    if c_list[3] == 1:
                        success_modifier_norm_index_3.append(index_success)
                    if c_list[4] == 1:
                        success_modifier_norm_index_4.append(index_success)
                    if c_list[5] == 1:
                        success_modifier_norm_index_5.append(index_success)
                    index_success = index_success + 1
                print('success:{}/{}'.format(success_count, index + 1))
                index = index + 1
            # if index == (args.uap_test_index_end - args.uap_test_index_start):
            if index == 30:
                break

    if args.app == 'none_target_attack':
        # none target attack
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in a['{}_{}'.format(args.label_difficult, args.k_value)]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()
                if index < 1:
                    print('\n')
                    data, GT = data.to(device), GT.to(device)

                    plot_list = []
                    plot_list.append(data.cpu().detach().numpy())
                    plot_list.append(GT.cpu().detach().numpy())
                    plot_list.append(np.argsort(-(model(data).cpu().detach().numpy())[0]))

                    purtabed_img_out, modifier_out, converge_iter, attack_success, c_list, purtabed_final_out, predict_label = \
                        l2_topk_non_targeted_attack(model, data, GT, args.k_value, args.eps, args.maxiter, args.boxmax,
                                                    args.boxmin, \
                                                    device, Projection_flag=True, lr=args.lr_attack)

                    plot_list.append(purtabed_img_out)
                    plot_list.append(purtabed_final_out)
                    plot_list.append(predict_label)
                    plot_list.append(modifier_out)
                    os.makedirs(
                        './plot_result/{}/{}/{}/eps_{}'.format(args.dataset, args.label_difficult, args.app, args.eps),
                        exist_ok=True)
                    plot_list = np.asarray(plot_list, dtype=object)
                    np.save(
                        './plot_result/{}/{}/{}/eps_{}/images_result_k_{}_ith_{}'.format(args.dataset,
                                                                                         args.label_difficult, args.app,
                                                                                         args.eps,
                                                                                         args.k_value, ith), plot_list)

                    if attack_success:
                        success_count = success_count + 1
                        # success_perturbated_img_list.append(purtabed_img_out)
                        # success_modifier_list.append(modifier_out)
                        success_modifier_norm_list.append(
                            np.linalg.norm(modifier_out) / ((args.image_size) * (args.image_size)))
                        success_img_index.append(ith)
                        success_c_list.append(c_list)
                        if c_list[0] == 1:
                            success_modifier_norm_index_0.append(index_success)
                        if c_list[1] == 1:
                            success_modifier_norm_index_1.append(index_success)
                        if c_list[2] == 1:
                            success_modifier_norm_index_2.append(index_success)
                        if c_list[3] == 1:
                            success_modifier_norm_index_3.append(index_success)
                        if c_list[4] == 1:
                            success_modifier_norm_index_4.append(index_success)
                        if c_list[5] == 1:
                            success_modifier_norm_index_5.append(index_success)
                        index_success = index_success + 1
                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == 1:
                break

    if args.app == 'target_attack':
        if args.dataset == "DMNIST" or args.dataset == "TMNIST" or args.dataset == "PASCAL-VOC":
            if args.dataset == "DMNIST" or args.dataset == "TMNIST":
                if args.attacked_label == '':
                    attacked_label = [int(f[-5]) for f in os.listdir(args.data)]
                else:
                    attacked_label = ast.literal_eval(args.attacked_label)  # this should be a list

            customized = ast.literal_eval(args.target_labels)
        else:
            attacked_label = []
            customized = []
        if args.sample_name == 'all':
            imgs_num = 50
        else:
            imgs_num = 1
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if args.dataset == "DMNIST" or args.dataset == "TMNIST" or args.dataset == "PASCAL-VOC" or ith in a['{}_{}'.format(args.label_difficult, args.k_value)]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                print('ith:', ith)
                GT = GT.int()
                if index < imgs_num:
                    # target attack
                    if args.dataset == "DMNIST" or args.dataset == "TMNIST":
                        al = attacked_label[index]
                    else:
                        al = 'none'
                    print('\n')
                    data = data.to(device)
                    target, targets_zeros = generate_target_zeros_3_cases(model, data, GT.cpu().detach().numpy(),
                                                                          args.label_difficult, k=args.k_value,
                                                                          customized=customized, attacked_label=al)
                    plot_list = []
                    plot_list.append(data.cpu().detach().numpy())
                    plot_list.append(GT.cpu().detach().numpy())
                    plot_list.append(np.argsort(-(model(data).cpu().detach().numpy())[0]))
                    plot_list.append(targets_zeros)

                    target = torch.from_numpy(target).to(device)
                    purtabed_img_out, modifier_out, attack_success, c_list, purtabed_final_out, predict_label = \
                        l2_topk_targeted_attack(model, data, target, args.k_value, args.eps, args.maxiter, args.boxmax,
                                                args.boxmin, \
                                                targets_zeros, GT, device=device,
                                                remove_tier_para=args.remove_tier_para, lr=args.lr_attack,
                                                norm=args.norm)

                    plot_list.append(purtabed_img_out)
                    plot_list.append(purtabed_final_out)
                    plot_list.append(predict_label)
                    plot_list.append(modifier_out)
                    if args.sample_name == "all":
                        os.makedirs(
                            './plot_result/{}/{}/{}/eps_{}/{}_norm/def_{}'.format(args.dataset, args.label_difficult,
                                                                                  args.app, args.eps, args.norm,
                                                                                  args.defense),
                            exist_ok=True)
                        plot_list = np.asarray(plot_list, dtype=object)
                        np.save(
                            './plot_result/{}/{}/{}/eps_{}/{}_norm/def_{}/images_result_k_{}_ith_{}'.format(
                                args.dataset, args.label_difficult, args.app, args.eps,
                                args.norm, args.defense, args.k_value, ith), plot_list)
                    else:
                        os.makedirs(
                            './plot_result_singles/{}/{}/{}/eps_{}/{}_norm/def_{}'.format(args.dataset, args.label_difficult,
                                                                                  args.app, args.eps, args.norm,
                                                                                  args.defense),
                            exist_ok=True)
                        plot_list = np.asarray(plot_list, dtype=object)
                        file_name, _ = os.path.splitext(args.sample_name)
                        np.save(
                            './plot_result_singles/{}/{}/{}/eps_{}/{}_norm/def_{}/image_result_k_{}_sample_{}_{}'.format(
                                args.dataset, args.label_difficult, args.app, args.eps,
                                args.norm, args.defense, args.k_value, file_name, args.target_labels)
                            , plot_list)

                    if attack_success:
                        success_count = success_count + 1
                        # success_perturbated_img_list.append(purtabed_img_out)
                        # success_modifier_list.append(modifier_out)
                        success_modifier_norm_list.append(
                            np.linalg.norm(modifier_out) / ((args.image_size) * (args.image_size)))
                        success_img_index.append(ith)
                        success_c_list.append(c_list)
                        if c_list[0] == 1:
                            success_modifier_norm_index_0.append(index_success)
                        if c_list[1] == 1:
                            success_modifier_norm_index_1.append(index_success)
                        if c_list[2] == 1:
                            success_modifier_norm_index_2.append(index_success)
                        if c_list[3] == 1:
                            success_modifier_norm_index_3.append(index_success)
                        if c_list[4] == 1:
                            success_modifier_norm_index_4.append(index_success)
                        if c_list[5] == 1:
                            success_modifier_norm_index_5.append(index_success)
                        index_success = index_success + 1
                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == imgs_num:
                break

    # save_result(success_c_list, index, args, success_img_index, success_modifier_norm_list,
    #             success_modifier_norm_index_0, success_modifier_norm_index_1, \
    #             success_modifier_norm_index_2, success_modifier_norm_index_3, \
    #             success_modifier_norm_index_4, success_modifier_norm_index_5)

    torch.cuda.empty_cache()
