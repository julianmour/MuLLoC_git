import os
import numpy as np
from PIL import Image

attacked_label = [int(f[-5]) for f in os.listdir("double_mnist_samples")]

for norm in ["l2", "lInf"]:
    for model in ['noDef', 'L0Def', 'LInfDef']:
        # succeeded = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]
        for i in range(30):
            # if i not in succeeded:
            #     continue
            array = \
                np.load(r"plot_result\DMNIST\customized\target_attack\eps_3\{}_norm\def_".format(norm)+model+r"\images_result_k_2_al_" + str(attacked_label[i]) +
                        "_ith_" + str(i) + ".npy", allow_pickle=True)
            # print(np.squeeze(array[4]))
            image = Image.fromarray(np.squeeze(array[0]) * 255).convert("L")
            perturbed = Image.fromarray(np.squeeze(array[4]) * 255).convert("L")
            perturbation = Image.fromarray((abs(np.squeeze(array[0]) - np.squeeze(array[4])) * 255)**2).convert("L")

            # Save the image
            image.save("plot_result\\DMNIST\\customized\\target_attack\\eps_3\\{}_norm\\def_".format(norm)+model+"\\image" + str(i) + ".png")
            perturbed.save("plot_result\\DMNIST\\customized\\target_attack\\eps_3\\{}_norm\\def_".format(norm)+model+"\\perturbed" + str(i) + ".png")
            perturbation.save("plot_result\\DMNIST\\customized\\target_attack\\eps_3\\{}_norm\\def_".format(norm)+model+"\\perturbation" + str(i) + ".png")
