import os
import subprocess
import time
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
import multiprocessing


def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
    

def attack_single_image(command, attack_result_path, attack_failed_path, perturbed_np_path):
    print("Attacking single image\n")
    output = run_command(command)
    attack_success = (output[-3] == "1")

    while True:
        if os.path.exists(attack_result_path):
            if attack_success:
                array = np.load(attack_result_path, allow_pickle=True)
                np.save(perturbed_np_path, np.squeeze(np.array(array[4])))

            else:
                f = open("tmp_"+attack_failed_path, "w")
                f.close()
                os.replace("tmp_"+attack_failed_path, attack_failed_path)
            break


if __name__ == '__main__':

    ready_path = "pyCommand_attacks_ready.txt"

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    while True:

        if os.path.exists(ready_path):

            os.remove(ready_path)  # delete text file

            files = glob.glob(f'{"pyCommand_attack"}*')

            for tf_path in files:
                f = open(tf_path, "r")
                cmd = f.readline()
                attack_result_path = f.readline()
                attack_failed_path = f.readline()
                perturbed_np_path = f.readline()
                f.close()
                os.remove(tf_path)  # delete text file
                process = multiprocessing.Process(target=attack_single_image, args=(cmd, attack_result_path, attack_failed_path, perturbed_np_path,))
                process.start()
