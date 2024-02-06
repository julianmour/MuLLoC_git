import os
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    disc = "colors represent |TkML-AP| - \u03B5 where [" + u"\u00B1" + "\u03B5] is the robust perturbation of a layer."
    models = ['noDef', 'L0Def', 'LInfDef']
    mask_types = ['target', 'non-target', 'all', 'pixel-independent', 'epsilon-ball']
    # mask_types = ['pixel-independent']
    valid_avg_sizes = {"fw": [], "sw": []}

    directory = r"C:\Users\julianmour\OneDrive - Technion\Documents\Research\double_mnist\samples"

    file_names = os.listdir(directory)
    img_idxes = [os.path.splitext(file_name)[0] for file_name in file_names]

    # succeeded = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]
    attacked_label = [int(f[-5]) for f in os.listdir("double_mnist_samples")]

    for mt in mask_types:
        print("generating for {} mask-type...".format(mt))
        for norm in ['l2', 'lInf']:
            for model_name in models:

                table = {
                    'Image': [],
                    'Minimal epsilon of attacked pixels': [],
                    'Number of attacked pixels': [],
                    'Minimal epsilon of attacked pixels outside our neighborhood': [],
                    '% of attacked pixels outside our neighborhood': [],
                    'Maximal robust epsilon-ball': [],
                    'Maximal epsilon in our neighborhood': [],
                    'Average epsilon in our neighborhood': []
                }
                directory = r"plot_result\DMNIST\customized\target_attack\eps_3\differences\{}_norm_attack\Heatmaps_fw_".format(
                    norm) + model_name + "_" + mt
                for ith, i in tqdm(enumerate(img_idxes), desc="Processing", ncols=100, total=len(img_idxes)):
                    # plt.style.use("seaborn")
                    file_path = r"C:\Users\julianmour\OneDrive - Technion\Documents" \
                                r"\Research\OptimalVerify\src\Results_fw_" + model_name + "_" + mt + r"\Image" + str(
                        i) + r"\heatmap_img.npy"
                    if not os.path.isfile(file_path):
                        # sizes.append(-1)
                        continue

                    data = np.load(file_path)

                    attack_list = \
                        np.load(
                            r"plot_result\DMNIST\customized\target_attack\eps_3\{}_norm\def_{}\images_result_k_2_al_{}_ith_{}.npy".
                                format(norm, model_name, str(attacked_label[ith]), str(ith)), allow_pickle=True)
                    attack = abs(np.squeeze((np.squeeze(attack_list[4]) - np.squeeze(attack_list[0]))))
                    original_data = [10 ** x for x in data]
                    diff = attack - original_data
                    # if np.array_equal(np.squeeze(attack_list[4]), np.squeeze(attack_list[0])):
                    #     continue  # unsuccessful attack
                    # MPD = np.min(diff[diff > 0])

                    cmap = sns.diverging_palette(120, 10, as_cmap=True)
                    ax = sns.heatmap(diff, cmap=cmap, center=0)
                    # cbar = plt.gca().collections[0].colorbar
                    # cbar.ax.axhline(MPD, color='black', linestyle='--')
                    plt.suptitle("Differences heatmap Image " + str(i) + " (fixed weights)", fontsize=14,
                                 fontweight='bold')
                    plt.title(disc, fontsize=10, style='italic')

                    # saving
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(directory + "\diff_heatmap_" + str(i) + ".png")
                    plt.close()

                    data_flattened = diff.flatten()

                    # Set the number of bins for the histogram
                    num_bins = 50

                    # Create the histogram
                    plt.hist(data_flattened, bins=num_bins, alpha=0.5, color='b', edgecolor='k')

                    # Add labels and title
                    plt.xlabel('|TkML-AP| - \u03B5')
                    plt.ylabel('Frequency')
                    plt.suptitle("Histogram of differences for Image " + str(i) + " (fixed weights)", fontsize=12,
                                 fontweight='bold')

                    # Save the histogram as an image file (e.g., PNG)
                    plt.savefig(directory + "\diff_histogram_" + str(i) + ".png")
                    plt.close()

                table = {
                    'Image': [],
                    'Minimal epsilon of attacked pixels': [],
                    'Number of attacked pixels': [],
                    'Minimal epsilon of attacked pixels outside our neighborhood': [],
                    '% of attacked pixels outside our neighborhood': [],
                    'Maximal robust epsilon-ball': [],
                    'Maximal epsilon in our neighborhood': [],
                    'Average epsilon in our neighborhood': [],
                }
                directory = r"plot_result\DMNIST\customized\target_attack\eps_3\differences\{}_norm_attack\Heatmaps_sw_".format(
                    norm) + model_name + "_" + mt
                table_directory = r"tables\{}_norm_attack\Heatmaps_sw_{}_{}".format(norm, model_name, mt)
                for ith, i in tqdm(enumerate(img_idxes), desc="Processing", ncols=100, total=len(img_idxes)):
                    # plt.style.use("seaborn")
                    file_path = r"C:\Users\julianmour\OneDrive - Technion\Documents" \
                                r"\Research\OptimalVerify\src\Results_sw_" + model_name + "_" + mt + "\Image" + str(i) + \
                                "\heatmap_img.npy"
                    if not os.path.isfile(file_path):
                        # sizes.append(-1)
                        continue

                    data = np.load(file_path)

                    attack_list = \
                        np.load(
                            r"plot_result\DMNIST\customized\target_attack\eps_3\{}_norm\def_{}\images_result_k_2_al_{}_ith_{}.npy".
                                format(norm, model_name, str(attacked_label[ith]), str(ith)), allow_pickle=True)
                    attack = abs(np.squeeze((np.squeeze(attack_list[4]) - np.squeeze(attack_list[0]))))
                    original_data = [10 ** x for x in data]
                    diff = attack - original_data

                    cmap = sns.diverging_palette(120, 10, as_cmap=True)
                    ax = sns.heatmap(diff, cmap=cmap, center=0)
                    plt.suptitle("Differences heatmap Image " + str(i) + " (sensitivity weights)", fontsize=14,
                                 fontweight='bold')
                    plt.title(disc, fontsize=10, style='italic')
                    # saving
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(directory + "\diff_heatmap_" + str(i) + ".png")
                    plt.close()

                    # add to table
                    max_eps_neighborhood = np.max(original_data)
                    avg_eps_neighborhood = sum(np.array(original_data).flatten()) / len(
                        np.array(original_data).flatten())
                    min_eps_attack = np.min(attack[attack > 0])
                    num_of_attacked_pixels = len(attack[attack > 0])
                    if len(attack[diff > 0]) != 0:
                        min_eps_attack_outside_neighborhood = np.min(attack[diff > 0])
                    else:
                        min_eps_attack_outside_neighborhood = "-"
                    num_of_attacked_pixels_outside_neighborhood = len(diff[diff > 0])
                    ratio_ = num_of_attacked_pixels_outside_neighborhood / num_of_attacked_pixels
                    eps_ball = 0  # todo

                    table['Image'].append(i)
                    table['Minimal epsilon of attacked pixels'].append(min_eps_attack)
                    table['Number of attacked pixels'].append(num_of_attacked_pixels)
                    table['Minimal epsilon of attacked pixels outside our neighborhood'].append(
                        min_eps_attack_outside_neighborhood)
                    table['% of attacked pixels outside our neighborhood'].append(ratio_)
                    table['Maximal robust epsilon-ball'].append(eps_ball)
                    table['Maximal epsilon in our neighborhood'].append(max_eps_neighborhood)
                    table['Average epsilon in our neighborhood'].append(avg_eps_neighborhood)

                    if not os.path.exists(table_directory):
                        os.makedirs(table_directory)

                    # build diff histogram
                    data_flattened = diff.flatten()
                    num_bins = 50

                    plt.hist(data_flattened, bins=num_bins, alpha=0.5, color='b', edgecolor='k')

                    plt.xlabel('|TkML-AP| - \u03B5')
                    plt.ylabel('Frequency')
                    plt.suptitle("Histogram of differences for Image " + str(i) + " (sensitivity weights)", fontsize=12,
                                 fontweight='bold')

                    plt.savefig(directory + "\diff_histogram_" + str(i) + ".png")
                    plt.close()

                    # build neighborhood+attack graphs
                    x = np.array(range(len(attack.flatten())))
                    y_attack = np.array(attack.flatten())  # attack function
                    y_neighborhood = np.array(np.array(original_data).flatten())    # neighborhood function

                    plt.figure(figsize=(8, 4))  # Set the figure size
                    plt.plot(x, y_attack, label='Attack', color='red')
                    plt.plot(x, y_neighborhood, label='Neighborhood', color='blue')

                    plt.xlabel('pixel')
                    plt.ylabel('epsilon')
                    plt.title("Attack vs Neighborhood for Image " + str(i) + " (sensitivity weights)")
                    plt.legend()

                    plt.savefig(directory + "\AttackVsNeighborhood_" + str(i) + ".png", bbox_inches='tight', pad_inches=0.2)
                    plt.close()

                if os.path.exists(table_directory):
                    # Convert the dictionary to a DataFrame (a table)
                    df = pd.DataFrame(table)

                    # Save the DataFrame to a CSV file
                    df.to_csv(table_directory + r'\Table_{}_{}-norm_{}_sw.csv'.format(model_name, norm, mt),
                              index=False)
