# # -*- codeing = utf-8 -*-
# # @Time : 2024/3/5 14:35
# # @Author : 李国锋
# # @File: miniMNISTset.py
# # @Softerware:
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset
# import numpy as np



import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import random_split

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

# 定义每个客户端的数据量
num_clients = 100
data_per_client = len(mnist_dataset) // num_clients

# 分配数据集
clients_data = random_split(mnist_dataset, [data_per_client] * num_clients)


__degree = 0.5
__clientnum = 100
__batsize = 32

# 定义自己的MNIST数据集类
class CustomMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, indices):
        self.mnist_dataset = mnist_dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.mnist_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

# 加载原始MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

# 使用Dirichlet分布生成non-iid数据索引
def dirichlet_distribution(labels, alpha, n_clients):
    label_distribution = np.random.dirichlet([alpha] * n_clients, size=len(np.unique(labels)))
    # print(label_distribution.shape)
    class_indices = [np.where(labels == i)[0] for i in range(len(np.unique(labels)))]
    client_indices = [[] for _ in range(n_clients)]
    for c, indices in zip(label_distribution, class_indices):
        for i, p in enumerate(c):
            client_indices[i].extend(indices[:int(p * len(indices))])
            indices = indices[int(p * len(indices)):]
    return client_indices

# 获取每个客户端的数据索引
client_indices = dirichlet_distribution(mnist_trainset.targets.numpy(), alpha=__degree, n_clients=__clientnum )

# 创建客户端的数据加载器
client_dataloaders = [DataLoader(CustomMNISTDataset(mnist_trainset, indices), batch_size=__batsize, shuffle=True) for indices in client_indices]


###划分数据集


# print(client_dataloaders[9])
# for i in client_dataloaders:
#     print(len(i))
# from collections import Counter
#
# # 假设您的DataLoader已经定义好了，名为'dataloader'
# # 并且您的数据集中的标签是整数类型
#
# class_counts = Counter()
#
# for _, labels in client_dataloaders[0]:
#     class_counts.update(labels.tolist())
#
# # 打印每个类别的个数
# for class_id, count in class_counts.items():
#     print(f'Class {class_id}: {count}')









# import torch
# from torchvision import datasets, transforms
# import numpy as np
#
# # 加载MNIST数据集
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#
#
# # 定义函数以根据Dirichlet分布生成non-iid数据索引
# def non_iid_split(dataset, num_clients, alpha):
#     # 获取数据集的标签
#     labels = dataset.targets.numpy()
#     num_classes = len(np.unique(labels))
#
#     # 使用Dirichlet分布来分配标签
#     label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
#
#     # 为每个客户端创建数据索引列表
#     client_indices = {i: np.array([], dtype='int64') for i in range(num_clients)}
#     for class_idx in range(num_classes):
#         class_indices = np.where(labels == class_idx)[0]
#         # 根据Dirichlet分布分配标签
#         class_splits = np.split(class_indices,
#                                 (np.cumsum(label_distribution[:, class_idx])[:-1] * len(class_indices)).astype(int))
#         for client_idx, client_class_indices in enumerate(class_splits):
#             client_indices[client_idx] = np.concatenate((client_indices[client_idx], client_class_indices))
#
#     return client_indices
#
#
# # 分配non-iid数据索引
# client_indices = non_iid_split(mnist_dataset, num_clients=10, alpha=0.6)
#
# # 检查分配结果
# for client_idx, indices in client_indices.items():
#     print(f'Client {client_idx}: Number of samples - {len(indices)}')



#
# class NonIIDMNIST(Dataset):
#     def __init__(self, mnist_dataset, user_indices):
#         self.mnist_dataset = mnist_dataset
#         self.user_indices = user_indices
#
#     def __len__(self):
#         return len(self.user_indices)
#
#     def __getitem__(self, index):
#         mnist_index = self.user_indices[index]
#         return self.mnist_dataset[mnist_index]
#
# def non_iid_split(mnist_dataset, num_users=100, degree_of_noniid=0.3):
#     num_items_per_user = int(len(mnist_dataset) / num_users)
#     all_indices = [i for i in range(len(mnist_dataset))]
#     user_indices = []
#
#     # Calculate the number of items per class per user
#     num_classes = 10
#     base_items_per_class = int((1 - degree_of_noniid) * num_items_per_user / num_classes)
#     noniid_items_per_class = num_items_per_user - base_items_per_class * num_classes
#
#     # Distribute data to users ensuring non-iid
#     for user_id in range(num_users):
#         # Get base distribution for all classes
#         user_class_distribution = [base_items_per_class] * num_classes
#         # Add non-iid portion to random classes
#         noniid_classes = np.random.choice(num_classes, noniid_items_per_class, replace=True)
#         for cls in noniid_classes:
#             user_class_distribution[cls] += 1
#
#         # Sample indices for the user
#         user_idx = []
#         for cls, num_items in enumerate(user_class_distribution):
#             cls_indices = np.where(np.array(mnist_dataset.targets) == cls)[0]
#             cls_indices = np.random.choice(cls_indices, num_items, replace=False)
#             user_idx.extend(cls_indices)
#
#         # Shuffle indices for the user
#         np.random.shuffle(user_idx)
#         user_indices.append(user_idx)
#
#     return user_indices
#
# # MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#
# # Split dataset
# user_indices = non_iid_split(mnist_trainset, degree_of_noniid=0.3)
#
# # Create a NonIIDMNIST instance for the first user
# first_user_dataset = NonIIDMNIST(mnist_trainset, user_indices[0])
# first_user_loader = DataLoader(first_user_dataset, batch_size=64, shuffle=True)
#
# # Now you can use first_user_loader in your training loop
