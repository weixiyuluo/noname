# -*- codeing = utf-8 -*-
# @Time : 2024/3/8 20:43
# @Author : 李国锋
# @File: miniClient.py
# @Softerware:
import random
import time
import torch
import numpy as np
import torchvision
from Model import *  #自己写的模型
from torch import nn
import torch.nn.functional  as F
from torch.utils.data import DataLoader
from miniMNISTset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")


class miniClient():

    def __init__(self, clientid=0,initmodel = None,lr=0.001,batsize = 32):
        self.batchsize = batsize

        # 创建网络模型
        self.model_my = CovModel().to(device)
        if initmodel is not None:
            parms = CovModel().state_dict().copy()
            for key,newvalue in zip(parms.keys(),initmodel):
                parms[key] = torch.from_numpy(newvalue)
            self.model_my.load_state_dict(parms)
            self.model_my.to(device)
            #self.model_my.load_state_dict(initmodel.state_dict())

        self.loss_fnc =  F.cross_entropy  #torch.nn.functional.cross_entropy()
        __learning_rate = lr
        self.optimizer = torch.optim.SGD(self.model_my.parameters(), lr=__learning_rate)#,momentum=0.9
        #self.optimizer = torch.optim.Adam(self.model_my.parameters(), lr=__learning_rate)  #
        self.id = clientid
        self.__train_data = clients_data[self.id]
        # self.epsilon = 2  # 差分隐私预算
        # self.delta = 0.00001
        # self.sigma = np.sqrt(2 * (np.log(1.25 / self.delta))) / self.epsilon

    # 训练
    def train(self, globalmodel = None, epoch=1,):

        __train_loader = DataLoader(self.__train_data, batch_size=self.batchsize, shuffle=True)

        if globalmodel is not None:
            parms = self.model_my.state_dict().copy()
            for key, newvalue in zip(parms.keys(), globalmodel):
                parms[key] = torch.from_numpy(newvalue)
            self.model_my.load_state_dict(parms)
            #self.model_my.to(device)
        else:
            print("can not find global model")
            exit(0)
        # T1 = time.time()
        for i in range(epoch):
            # model_len  = sum(p.numel() for p in self.model_my.parameters() if p.requires_grad)
            # global_grad = np.zeros(model_len)
            # layers = np.zeros(0)
            total_acc = 0  # 每一个round的准确率
            num = 0
            for data in __train_loader:
                imgs, target = data
                imgs = imgs.to(device)
                target = target.to(device)
                output = self.model_my(imgs) + 1e-9
                self.optimizer.zero_grad()
                loss = self.loss_fnc(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_my.parameters(), max_norm=5)
                self.optimizer.step()
                pre_output = output.argmax(1)
                accuracy = (pre_output == target).sum()
                total_acc = total_acc + accuracy
                # num+=1
                # if num>=5:
                #     break

                # for name, param in self.model_my.named_parameters():  # self.model_my.model._modules.
                #     if param.requires_grad:
                #         layers = np.concatenate((layers, param.grad.cpu().flatten()),
                #                                 axis=None)  # 将梯度展开为1维数组#GPU训练的梯度要先放到cpu中才能与np运算
                #         global_grad +=layers

            print(f"client {self.id} train accuracy is {total_acc /len(self.__train_data)}")
            # T2 = time.time()
            # print("训练时间：",T2-T1)
        updateparm = []
        par = self.model_my.state_dict().copy()
        for key in par.keys():
            updateparm.append(par[key].cpu().numpy())
        updateparm = np.array(updateparm,dtype=object)
        grad =  globalmodel - updateparm
        # noise = self.sigma* np.random.randn(model_len)
        # grad = global_grad + noise             #添加差分隐私

        # return grad, len(__train_loader)
        return grad



    def myTest(self, model, signal = 0):
        if(signal == 1):
            parms = self.model_my.state_dict().copy()
            for key, newvalue in zip(parms.keys(), model):
                parms[key] = torch.from_numpy(newvalue)
            tesrmodel = CovModel().to(device)
            tesrmodel.load_state_dict(parms)
        else:
            tesrmodel = model
        __test_data = torchvision.datasets.MNIST("../data/MNIST", train=False,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ]),
                                               download=True)
        __test_load = DataLoader(__test_data, batch_size=128)
        test_data_size = len(__test_data)
        tol_acc = 0
        tesrmodel.eval()
        for data in __test_load:
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)
            output = tesrmodel(imgs)
            pre_label = output.argmax(1)
            acc = (pre_label == label).sum()
            tol_acc += acc
        # torch.save(model, "model_test")
        print(f"client{self.id} test accuracy is{tol_acc / test_data_size}")
        return tol_acc / test_data_size
        # for name, param in self.model_my.named_parameters():  # self.model_my.model._modules.
        #     if param.requires_grad:
        #         print(param.data)


if __name__ == '__main__':

    m = CovModel().to(device)
    model = []
    par = m.state_dict().copy()
    for key in par.keys():
        model.append(par[key].cpu().numpy())
    model = np.array(model,dtype=object)
    c = miniClient(initmodel=model,clientid=0)
    grad = c.train(globalmodel=model,epoch=10)
    # print()
    newmodel = model - grad
    c.myTest(newmodel,1)
    #
    c.train(newmodel)



# class noniidClient():
#
#     def __init__(self, clientid=0,initmodel = None,lr=0.01,batsize = 32):
#         self.batchsize = batsize
#
#         # 创建网络模型
#         self.model_my = CovModel().to(device)
#         if initmodel is not None:
#             parms = CovModel().state_dict().copy()
#             for key,newvalue in zip(parms.keys(),initmodel):
#                 parms[key] = torch.from_numpy(newvalue)
#             self.model_my.load_state_dict(parms)
#             self.model_my.to(device)
#             #self.model_my.load_state_dict(initmodel.state_dict())
#
#         self.loss_fnc =  F.cross_entropy  #torch.nn.functional.cross_entropy()
#         __learning_rate = lr
#         self.optimizer = torch.optim.SGD(self.model_my.parameters(), lr=__learning_rate)#,momentum=0.9
#         #self.optimizer = torch.optim.Adam(self.model_my.parameters(), lr=__learning_rate)  #
#         self.id = clientid
#         # self.epsilon = 2  # 差分隐私预算
#         # self.delta = 0.00001
#         # self.sigma = np.sqrt(2 * (np.log(1.25 / self.delta))) / self.epsilon
#
#     # 训练
#     def train(self, globalmodel = None, epoch=1,):
#
#         __train_loader = client_dataloaders[self.id]
#         if globalmodel is not None:
#             parms = self.model_my.state_dict().copy()
#             for key, newvalue in zip(parms.keys(), globalmodel):
#                 parms[key] = torch.from_numpy(newvalue)
#             self.model_my.load_state_dict(parms)
#             #self.model_my.to(device)
#         else:
#             print("can not find global model")
#             exit(0)
#         # T1 = time.time()
#         for i in range(epoch):
#             # model_len  = sum(p.numel() for p in self.model_my.parameters() if p.requires_grad)
#             # global_grad = np.zeros(model_len)
#             # layers = np.zeros(0)
#             total_acc = 0  # 每一个round的准确率
#             for data in __train_loader:
#                 imgs, target = data
#                 imgs = imgs.to(device)
#                 target = target.to(device)
#                 output = self.model_my(imgs) + 1e-9
#                 self.optimizer.zero_grad()
#                 loss = self.loss_fnc(output, target)
#                 loss.backward()
#                 self.optimizer.step()
#                 pre_output = output.argmax(1)
#                 accuracy = (pre_output == target).sum()
#                 total_acc = total_acc + accuracy
#
#
#                 # for name, param in self.model_my.named_parameters():  # self.model_my.model._modules.
#                 #     if param.requires_grad:
#                 #         layers = np.concatenate((layers, param.grad.cpu().flatten()),
#                 #                                 axis=None)  # 将梯度展开为1维数组#GPU训练的梯度要先放到cpu中才能与np运算
#                 #         global_grad +=layers
#
#             print(f"client {self.id} train accuracy is {total_acc / (len(__train_loader) * self.batchsize)}")
#             # T2 = time.time()
#             # print("训练时间：",T2-T1)
#         updateparm = []
#         par = self.model_my.state_dict().copy()
#         for key in par.keys():
#             updateparm.append(par[key].cpu().numpy())
#         updateparm = np.array(updateparm,dtype=object)
#         grad =  globalmodel - updateparm
#         # noise = self.sigma* np.random.randn(model_len)
#         # grad = global_grad + noise             #添加差分隐私
#
#         # return grad, len(__train_loader)
#         return grad
#
#
#
#     def myTest(self, model, signal = 0):
#         if(signal == 1):
#             parms = self.model_my.state_dict().copy()
#             for key, newvalue in zip(parms.keys(), model):
#                 parms[key] = torch.from_numpy(newvalue)
#             tesrmodel = CovModel().to(device)
#             tesrmodel.load_state_dict(parms)
#         else:
#             tesrmodel = model
#         __test_data = torchvision.datasets.MNIST("../data/MNIST", train=False,
#                                                transform=torchvision.transforms.Compose([
#                                                    torchvision.transforms.ToTensor(),
#                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
#                                                ]),
#                                                download=True)
#         __test_load = DataLoader(__test_data, batch_size=128)
#         test_data_size = len(__test_data)
#         tol_acc = 0
#         tesrmodel.eval()
#         for data in __test_load:
#             imgs, label = data
#             imgs = imgs.to(device)
#             label = label.to(device)
#             output = tesrmodel(imgs)
#             pre_label = output.argmax(1)
#             acc = (pre_label == label).sum()
#             tol_acc += acc
#         # torch.save(model, "model_test")
#         print(f"client{self.id} test accuracy is{tol_acc / test_data_size}")
#         return tol_acc / test_data_size
#         # for name, param in self.model_my.named_parameters():  # self.model_my.model._modules.
#         #     if param.requires_grad:
#         #         print(param.data)
#

