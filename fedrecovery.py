import copy
import torch
import math
import numpy as np

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda:0")


test_ds = torchvision.datasets.MNIST('/home/joshua/UTS/Hanyu/LADProject/data', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))
                                     ]))



def test(network, data_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)
    test_loss /= len(data_loader.dataset)
    return accuracy, test_loss


# def gaussian_mechanism(s, d, eps, beta):
#     sigma = d/(np.sqrt(np.log(1/beta)+eps)-np.sqrt(np.log(1/beta)))/np.sqrt(2)
#     # std = (2 * math.log(1.25 / delta)) ** 0.5 * f / eps
#     noise = torch.normal(mean=0.0, std=sigma, size=s.size()).to(device)
#     return s+noise


def gaussian_mechanism(s, d, eps, beta):
    sigma = d / (np.sqrt(np.log(1 / beta) + eps) - np.sqrt(np.log(1 / beta))) / np.sqrt(2)
    # sigma = (2 * math.log(1.25 / beta)) ** 0.5 * d / eps
    for k, v in s.named_parameters():
        noise = torch.normal(mean=0.0, std=sigma, size=v.size()).to(device)
        v.data += noise
    # std = (2 * math.log(1.25 / delta)) ** 0.5 * f / eps
    # return s+noise


def distance(m1, m2):
    ss = 0
    for k in m1.keys():
        ss += (m1[k]-m2[k]).norm(2)**2
        # ss += ((m1[k] - m2[k]) ** 2).sum()
    return ss ** .5


# horizontal
test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)

# vertical
# d_k = np.where(test_ds.targets == 0)[0]
# test_loader = DataLoader(Subset(test_ds, d_k), batch_size=64, shuffle=True)

# 1&7
# d_3 = np.where(test_ds.targets == 0)[0].tolist()
# d_4 = np.where(test_ds.targets == 1)[0].tolist()
# test_loader = DataLoader(Subset(test_ds, d_3+d_4), batch_size=64, shuffle=True)

# load models
initial_model = Net().to(device)
initial_model.load_state_dict(torch.load('/home/joshua/UTS/Hanyu/LADProject/initial_model.pth'))

trained_model = Net().to(device)
trained_model.load_state_dict(torch.load('/home/joshua/UTS/Hanyu/LADProject/nn_model.pth'))

unlearned_model = Net().to(device)
unlearned_model.load_state_dict(torch.load('/home/joshua/UTS/Hanyu/LADProject/nn_model.pth'))

retrained_model = Net().to(device)
retrained_model.load_state_dict(torch.load('/home/joshua/UTS/Hanyu/LADProject/nn_model_1.pth'))


# load cache
dt = torch.load('/home/joshua/UTS/Hanyu/LADProject/nn_dt.pth')
# dt = torch.load('/home/joshua/UTS/Hanyu/LADProject/nn_lst.pth')
x1 = np.array(torch.load('/home/joshua/UTS/Hanyu/LADProject/ag_cache.pth'))
arg_x1 = x1.argsort()[-100:]
p=x1[arg_x1]
# p = x1/sum(x1)
# p=[1]*800
# print(sorted(p,reverse=True)[:10])


iteration = len(p)
# unlearning
for k, v in unlearned_model.named_parameters():
    dt[0][k] = dt[0][k]*p[0]
    for i in range(1, iteration):
        dt[0][k] += dt[i][k]*p[i]
    v.data = v.data - dt[0][k]

# compute distance
_, L1 = test(initial_model, test_loader)
print(_)
_, L2 = test(trained_model, test_loader)
print(_)
_, L3 = test(retrained_model, test_loader)
print(_)

learning_rate = 0.01
Dt = distance(initial_model.state_dict(), trained_model.state_dict())
d = Dt + np.sqrt(2*iteration*learning_rate*abs(L1-L2))


# add noise
eps = 500
beta = 0.1#0.2, 0.3, 0.4

gaussian_mechanism(retrained_model, d=d, eps=eps, beta=beta)
gaussian_mechanism(unlearned_model, d=d, eps=eps, beta=beta)

# test
acc1, _ = test(unlearned_model, test_loader)
acc2, _ = test(retrained_model, test_loader)

print(acc1)
print(acc2)

######################

# gaussian_mechanism(lst_model, d=d, eps=eps, beta=beta)
# acc_test_1, _ = test(lst_model, test_loader)
# print(acc_test_1)
