import torch.nn.functional as func
from torch import nn
import torch
import numpy as np

BATCH_SIZE = 4


class CNN_dmnist(nn.Module):
    def __init__(self):
        super().__init__()

        # image-tensor goes in as batch_sizex1x64x64

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)

        # image-tensor is batch_sizex8x32x32

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

        # image-tensor is batch_sizex16x16x16

        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)

        # image-tensor is batch_sizex16x4x4

        # flatten image tensor to batch_sizex16x4x4

        self.fc1 = nn.Linear(256, 10)  # 16 * 4 * 4 = 256

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, 256)  # 16 * 4 * 4 = 256
        x = func.softmax(self.fc1(x))

        return x

    def compute_gradient_to_input(self, x, label, minus_second_max=False):
        # define forward pass
        # x = x.view(-1, 64*64)
        image_for_gradient = np.copy(np.array(x))
        data = torch.from_numpy((np.array([image_for_gradient]))).float()
        data.requires_grad = True
        output = self.forward(data)
        indices = (-output[0].cpu().detach().numpy()).argsort()  # [:2]
        count = 0
        if minus_second_max:
            for ind_ in indices:
                if ind_ != label:
                    if count == 0:
                        count = count + 1
                    else:
                        loss = output[0][label] - output[0][ind_]
                        break
        else:
            loss = output[0][label]

        self.zero_grad()
        loss.backward()
        data_grad = data.grad.data.cpu().detach().numpy()
        if np.sum(data_grad) == 0:
            print("sum is zero!!")
        print(data_grad)
        return data_grad


class CNN_tmnist(nn.Module):
    def __init__(self):
        super().__init__()

        # image-tensor goes in as batch_sizex1x84x84

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)

        # image-tensor is batch_sizex8x42x42

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

        # image-tensor is batch_sizex16x21x21

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=3)

        # image-tensor is batch_sizex16x7x7

        # flatten image tensor to batch_sizex16x7x7

        self.fc1 = nn.Linear(784, 10)  # 16 * 7 * 7 = 784

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, 784)  # 16 * 7 * 7 = 784
        x = func.softmax(self.fc1(x))

        return x

    def compute_gradient_to_input(self, x, label, minus_second_max=False):
        # define forward pass
        # x = x.view(-1, 64*64)
        image_for_gradient = np.copy(np.array(x))
        data = torch.from_numpy((np.array([image_for_gradient]))).float()
        data.requires_grad = True
        output = self.forward(data)
        indices = (-output[0].cpu().detach().numpy()).argsort()  # [:2]
        count = 0
        if minus_second_max:
            for ind_ in indices:
                if ind_ != label:
                    if count == 0:
                        count = count + 1
                    else:
                        loss = output[0][label] - output[0][ind_]
                        break
        else:
            loss = output[0][label]

        self.zero_grad()
        loss.backward()
        data_grad = data.grad.data.cpu().detach().numpy()
        if np.sum(data_grad) == 0:
            print("sum is zero!!")
        print(data_grad)
        return data_grad

