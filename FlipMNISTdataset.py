
import os

import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import mnist


class NewMNIST(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, train=True, start=0, end=-1, poison=False, all = False,non_iid = False):
        super(NewMNIST, self).__init__()
        if train:
            dataset = mnist.MNIST('../data/MNIST', train=True, download=True)
        else:
            dataset = mnist.MNIST('../data/MNIST', train=False, download=True)
        if end == -1:
            self.images, self.labels = dataset.data.numpy(), dataset.targets.numpy()
        else:
            self.images, self.labels = dataset.data.numpy()[start: end], dataset.targets.numpy()[start: end]
        if poison:
            if train:
                self.poisoned_indices = np.where((self.labels == 8) | (self.labels == 1) | (self.labels == 2))[0]
                self.labels[np.where(self.labels == 8)] = 0
                self.labels[np.where(self.labels == 1)] = 7
                self.labels[np.where(self.labels == 2)] = 5
            else:
                self.poisoned_indices = np.where((self.labels == 8) | (self.labels == 1) | (self.labels == 2))[0]
                self.labels[np.where(self.labels == 8)] = 0
                self.labels[np.where(self.labels == 1)] = 7
                self.labels[np.where(self.labels == 2)] = 5
                if all:  #构建全是投毒数据的数据集
                    self.images = self.images[self.poisoned_indices]
                    self.labels = self.labels[self.poisoned_indices]
        # self.max_nums = len(dataset.data)
        # if non_iid:
        #     # 假设我们想要每个数字的样本数量不同
        #     # 例如，我们想要更多的 '1' 和 '7'，较少的 '0' 和 '5'
        #     desired_distribution = {0: 100, 1: 1000, 2: 200, 3: 300, 4: 400,
        #                             5: 100, 6: 600, 7: 1000, 8: 800, 9: 900}
        #     indices = np.hstack([np.random.choice(np.where(self.labels == i)[0],
        #                                           desired_distribution[i], replace=False)
        #                          for i in range(10)])
        #     self.images = self.images[indices]
        #     self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.images[index]
        x = np.array(x, dtype='float32') / 255
        x = (x - 0.5) / 0.5  # normalization    #转化为-1到1
        # x = x.reshape((-1,))  # flatten  #拉成一行,DNN用  维度转化  #(784,1)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = transform(Image.fromarray(x))
        label = self.labels[index]
        label = int(label)
        return image, label

    # def get_max_nums(self):
    #     return self.max_nums

    def get_poision_index(self):
        return self.poisoned_indices


if __name__=="__main__":
    c = NewMNIST(poison=True)
    # print(c[c.get_poision_num()])





# def trim_attack(data_loader, percentage=0.1):
#     """
#     对 DataLoader 中的数据进行 Trim attack。
#     :param data_loader: 包含数据的 DataLoader。
#     :param percentage: 被污染数据的比例。
#     :return: 新的 DataLoader，包含了部分被污染的数据。
#     """
#     # 计算需要污染的数据量
#     num_samples = len(data_loader.dataset)
#     num_poisoned = int(num_samples * percentage)
#
#     # 随机选择数据进行污染
#     poisoned_indices = np.random.choice(num_samples, num_poisoned, replace=False)
#     all_indices = set(range(num_samples))
#     clean_indices = list(all_indices - set(poisoned_indices))
#
#     # 创建包含污染数据的新 DataLoader
#     poisoned_loader = DataLoader(Subset(data_loader.dataset, poisoned_indices), batch_size=data_loader.batch_size, shuffle=True)
#     clean_loader = DataLoader(Subset(data_loader.dataset, clean_indices), batch_size=data_loader.batch_size, shuffle=True)
#
#     return poisoned_loader, clean_loader


# import torch
# import torchvision
# import numpy as np
#
# # 加载MNIST数据集
# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# # 选择要攻击的图像数量和攻击比例
# num_attack_samples = 100
# attack_ratio = 0.3
#
# # 选择要攻击的图像索引
# attack_indices = np.random.choice(len(train_dataset), size=num_attack_samples, replace=False)
# attack_data = [train_dataset[i] for i in attack_indices]
#
# # 对图像进行Trim攻击
# def trim_attack(image, epsilon=0.1):
#     noisy_image = image + epsilon * torch.randn_like(image)
#     return torch.clamp(noisy_image, 0, 1)     #将图片限制在【0，1】之间 #应该是要现正则化.#
#                                               将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。
#
# for i in range(len(attack_data)):
#     image, label = attack_data[i]
#     attack_data[i] = (trim_attack(image), label)
#
# # 将攻击后的图像混合回训练集
# for i in range(len(attack_data)):
#     train_dataset[attack_indices[i]] = attack_data[i]
#
# # 现在可以使用带有攻击数据的train_loader来训练模型


