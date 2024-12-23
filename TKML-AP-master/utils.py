# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:52:33 2019
@author: Keshik
"""
import os
import math
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import pandas as pd
import random
from PIL import Image
import torchvision.transforms as transforms
import pickle

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def generate_target_zeros(label, k=0):
    targets_zeros = np.zeros(label.shape, dtype=int)
    targets = np.full(label.shape, -1, dtype=int)
    targeted_list = []
    index = 0
    seq = range(label.shape[1])

    for j in seq:
        if (label[0][j] == 1) and (index < k):
            continue
        targeted_list.append(j)
        index = index + 1
        if index == k:
            break
    targets[0][targeted_list] = 1
    targets_zeros[0][targeted_list] = 1
    return targets, targets_zeros


def generate_target_zeros_3_cases(model, input, label, label_difficult, k=0, customized=None, attacked_label=0):
    if customized is None:
        customized = []
    predict = model(input).cpu().detach().numpy()
    targets_zeros = np.zeros(label.shape, dtype=int)
    targets = np.full(label.shape, -1, dtype=int)
    targeted_list = []
    index = 0

    GT_label = set(np.transpose(np.argwhere(label[0] == 1))[0])
    predict_index_ascend = np.argsort(predict[0])
    predict_index_descend = np.argsort(-predict[0])
    all_list = set(list(range(0, label.shape[1])))
    complement_set = all_list - GT_label

    if label_difficult == 'best':
        for j in predict_index_descend:
            if (j in GT_label) and (index < k):
                continue
            targeted_list.append(j)
            index = index + 1
            if index == k:
                break
    elif label_difficult == 'random':
        random.seed(0)
        targeted_list = random.sample(complement_set, k)
    elif label_difficult == 'worst':
        for j in predict_index_ascend:
            if (j in GT_label) and (index < k):
                continue
            targeted_list.append(j)
            index = index + 1
            if index == k:
                break
    elif label_difficult == 'customized':
        targeted_list = customized
    elif label_difficult == 'one':
        for j in predict_index_descend:
            if (j == attacked_label) and (index < k):
                continue
            targeted_list.append(j)
            index = index + 1
            if index == k:
                break

    print("target prediction = {}".format(str(targeted_list)))
    targets[0][targeted_list] = 1
    targets_zeros[0][targeted_list] = 1
    return targets, targets_zeros


def get_categories(labels_dir):
    """
    Get the object categories

    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a list
    """

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError

    else:
        categories = []

        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])

        return categories


def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding

    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """

    ls = target['annotation']['object']

    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))

    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))

    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k)


def get_nrows(file_name):
    """
    Get the number of rows of a csv file

    Args:
        file_path: path of the csv file
    Raises:
        FileNotFoundError: If the csv file does not exist
    Returns:
        number of rows
    """

    if not os.path.isfile(file_name):
        raise FileNotFoundError

    s = 0
    with open(file_name) as f:
        s = sum(1 for line in f)
    return s


def get_mean_and_std(dataloader):
    """
    Get the mean and std of a 3-channel image dataset

    Args:
        dataloader: pytorch dataloader
    Returns:
        mean and std of the dataset
    """
    mean = []
    std = []

    total = 0
    r_running, g_running, b_running = 0, 0, 0
    r2_running, g2_running, b2_running = 0, 0, 0

    with torch.no_grad():
        for data, target in tqdm(dataloader):
            r, g, b = data[:, 0, :, :], data[:, 1, :, :], data[:, 2, :, :]
            r2, g2, b2 = r ** 2, g ** 2, b ** 2

            # Sum up values to find mean
            r_running += r.sum().item()
            g_running += g.sum().item()
            b_running += b.sum().item()

            # Sum up squared values to find standard deviation
            r2_running += r2.sum().item()
            g2_running += g2.sum().item()
            b2_running += b2.sum().item()

            total += data.size(0) * data.size(2) * data.size(3)

    # Append the mean values
    mean.extend([r_running / total,
                 g_running / total,
                 b_running / total])

    # Calculate standard deviation and append
    std.extend([
        math.sqrt((r2_running / total) - mean[0] ** 2),
        math.sqrt((g2_running / total) - mean[1] ** 2),
        math.sqrt((b2_running / total) - mean[2] ** 2)
    ])

    return mean, std


def plot_history(train_hist, val_hist, y_label, filename, labels=["train", "validation"]):
    """
    Plot training and validation history

    Args:
        train_hist: numpy array consisting of train history values (loss/ accuracy metrics)
        valid_hist: numpy array consisting of validation history values (loss/ accuracy metrics)
        y_label: label for y_axis
        filename: filename to store the resulting plot
        labels: legend for the plot

    Returns:
        None
    """
    # Plot loss and accuracy
    xi = [i for i in range(0, len(train_hist), 2)]
    plt.plot(train_hist, label=labels[0])
    plt.plot(val_hist, label=labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()
    plt.close()


def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays

    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores


def save_results(images, scores, columns, filename):
    """
    Save inference results as csv

    Args:
        images: inferred image list
        scores: confidence score for inferred images
        columns: object categories
        filename: name and location to save resulting csv
    """
    df_scores = pd.DataFrame(scores, columns=columns)
    df_scores['image'] = images
    df_scores.set_index('image', inplace=True)
    df_scores.to_csv(filename)


def append_gt(gt_csv_path, scores_csv_path, store_filename):
    """
    Append ground truth to confidence score csv

    Args:
        gt_csv_path: Ground truth csv location
        scores_csv_path: Confidence scores csv path
        store_filename: name and location to save resulting csv
    """
    gt_df = pd.read_csv(gt_csv_path)
    scores_df = pd.read_csv(scores_csv_path)

    gt_label_list = []
    for index, row in gt_df.iterrows():
        arr = np.array(gt_df.iloc[index, 1:], dtype=int)
        target_idx = np.ravel(np.where(arr == 1))
        j = [object_categories[i] for i in target_idx]
        gt_label_list.append(j)

    scores_df.insert(1, "gt", gt_label_list)
    scores_df.to_csv(store_filename, index=False)


def get_classification_accuracy(gt_csv_path, scores_csv_path, store_filename):
    """
    Plot mean tail accuracy across all classes for threshold values

    Args:
        gt_csv_path: Ground truth csv location
        scores_csv_path: Confidence scores csv path
        store_filename: name and location to save resulting plot
    """
    gt_df = pd.read_csv(gt_csv_path)
    scores_df = pd.read_csv(scores_csv_path)

    # Get the top-50 images
    top_num = 2800
    image_num = 2
    num_threshold = 10
    results = []

    for image_num in range(1, 21):
        clf = np.sort(np.array(scores_df.iloc[:, image_num], dtype=float))[-top_num:]
        ls = np.linspace(0.0, 1.0, num=num_threshold)

        class_results = []
        for i in ls:
            clf = np.sort(np.array(scores_df.iloc[:, image_num], dtype=float))[-top_num:]
            clf_ind = np.argsort(np.array(scores_df.iloc[:, image_num], dtype=float))[-top_num:]

            # Read ground truth
            gt = np.sort(np.array(gt_df.iloc[:, image_num], dtype=int))

            # Now get the ground truth corresponding to top-50 scores
            gt = gt[clf_ind]
            clf[clf >= i] = 1
            clf[clf < i] = 0

            score = accuracy_score(y_true=gt, y_pred=clf, normalize=False) / clf.shape[0]
            class_results.append(score)

        results.append(class_results)

    results = np.asarray(results)

    ls = np.linspace(0.0, 1.0, num=num_threshold)
    plt.plot(ls, results.mean(0))
    plt.title("Mean Tail Accuracy vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Mean Tail Accuracy")
    plt.savefig(store_filename)
    plt.show()


def get_dmnist_samples(data_dir, device, sample_name='all'):
    data = []
    if sample_name == 'all':
        if not os.path.exists("all_dmnist_samples"):
            files = os.listdir(data_dir)
            for f in files:
                target = f[-6:-4]
                img_tensor = get_image_as_tensor(os.path.join(data_dir, f)).to(device)
                data.append((img_tensor, str_to_vector(target).to(device)))

            dump_file = open("all_dmnist_samples", 'wb')
            pickle.dump(data, dump_file)
            dump_file.close()
        load_file = open("all_dmnist_samples", 'rb')
    else:
        file_name, file_extension = os.path.splitext(sample_name)
        if not os.path.exists(file_name + "_dmnist_sample"):
            f = sample_name
            target = f[-6:-4]
            img_tensor = get_image_as_tensor(os.path.join(data_dir, f)).to(device)
            data.append((img_tensor, str_to_vector(target).to(device)))

            dump_file = open(file_name + "_dmnist_sample", 'wb')
            pickle.dump(data, dump_file)
            dump_file.close()
        load_file = open(file_name + "_dmnist_sample", 'rb')

    return pickle.load(load_file)


def get_tmnist_samples(data_dir, device, sample_name='all'):
    data = []
    if sample_name == 'all':
        if not os.path.exists("all_tmnist_samples"):
            files = os.listdir(data_dir)
            for f in files:
                target = f[-7:-4]
                img_tensor = get_image_as_tensor(os.path.join(data_dir, f)).to(device)
                data.append((img_tensor, str_to_vector(target).to(device)))

            dump_file = open("all_tmnist_samples", 'wb')
            pickle.dump(data, dump_file)
            dump_file.close()
        load_file = open("all_tmnist_samples", 'rb')
    else:
        file_name, file_extension = os.path.splitext(sample_name)
        if not os.path.exists(file_name + "_tmnist_sample"):
            f = sample_name
            target = f[-7:-4]
            img_tensor = get_image_as_tensor(os.path.join(data_dir, f)).to(device)
            data.append((img_tensor, str_to_vector(target).to(device)))

            dump_file = open(file_name + "_tmnist_sample", 'wb')
            pickle.dump(data, dump_file)
            dump_file.close()
        load_file = open(file_name + "_tmnist_sample", 'rb')

    return pickle.load(load_file)


def get_pascalvoc_samples(data_dir, device, sample_name='all'):
    data = []
    if sample_name == 'all':
        if not os.path.exists("all_pascalvoc_samples"):
            files = os.listdir(data_dir)
            for f in files:
                target = f[0:-5].split("_")[1]
                img_tensor = get_image_as_tensor(os.path.join(data_dir, f), pascal=True).to(device)
                data.append((img_tensor, str_to_vector_pascal(target).to(device)))

            dump_file = open("all_pascalvoc_samples", 'wb')
            pickle.dump(data, dump_file)
            dump_file.close()
        load_file = open("all_pascalvoc_samples", 'rb')
    else:
        file_name, file_extension = os.path.splitext(sample_name)
        if not os.path.exists(file_name + "_pascalvoc_sample"):
            f = sample_name
            target = f[0:-5].split("_")[1]
            img_tensor = get_image_as_tensor(os.path.join(data_dir, f), pascal=True).to(device)
            data.append((img_tensor, str_to_vector_pascal(target).to(device)))

            dump_file = open(file_name + "_pascalvoc_sample", 'wb')
            pickle.dump(data, dump_file)
            dump_file.close()
        load_file = open(file_name + "_pascalvoc_sample", 'rb')

    return pickle.load(load_file)


def get_image_as_tensor(img, pascal=False):
    # Read a PIL image
    image = Image.open(img)

    # Define a transform to convert PIL
    # image to a Torch tensor

    if pascal:
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    # return the converted Torch tensor
    return img_tensor


def str_to_vector(target):
    labels = torch.zeros([10], dtype=torch.float64)
    for c in target:
        labels[int(c)] = 1.
    return labels


def str_to_vector_pascal(target):
    labels = torch.zeros([20], dtype=torch.float64)
    classes = target.split("-")
    for c in classes:
        labels[int(c)] = 1.
    return labels

# get_classification_accuracy("../models/resnet18/results.csv", "../models/resnet18/gt.csv", "roc-curve.png")
