import os
import subprocess
import time
from PIL import Image
import numpy as np
from torchvision import transforms

# from CNN_model_4k import *


def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"


if __name__ == '__main__':

    tf_path = "pyCommand_attack.txt"
    attack_failed_path = "attack_failed.txt"
    perturbed_path = "perturbed_attack.png"
    perturbed_np_path = "perturbed_attack.npy"

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    while True:

        if os.path.exists(tf_path):
            print("Attacking single image\n")
            time.sleep(2)
            f = open(tf_path, "r")
            cmd = f.readline()
            attack_result_path = f.readline()
            f.close()
            os.remove(tf_path)  # delete text file
            output = run_command(cmd)
            print(output)
            attack_success = (output[-3] == "1")

            while True:
                if os.path.exists(attack_result_path):
                    if attack_success:
                        array = np.load(attack_result_path, allow_pickle=True)
                        # image = Image.fromarray(np.squeeze(array[0]) * 255).convert("L")
                        perturbed = Image.fromarray(np.squeeze(np.array(array[4])) * 255).convert("L")
                        # perturbation = Image.fromarray(
                        #     (abs(np.squeeze(array[0]) - np.squeeze(np.array(array[4]))) * 255) ** 2).convert("L")

                        # Save the perturbed image
                        # perturbed.save(perturbed_path)
                        np.save(perturbed_np_path, np.squeeze(np.array(array[4])))

                    else:
                        f = open(attack_failed_path, "w")
                        f.close()
                    break
