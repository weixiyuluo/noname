# -*- codeing = utf-8 -*-
# @Time : 2024/3/4 18:47
# @Author : 李国锋
# @File: backdoorMNISTClient.py
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
from backdoorMNISTset import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
seed = 3047
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



class backdoorClient():

    def __init__(self, clientid=0,initmodel = None,lr=0.001):
        self.batchsize = 128

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
        self.id = clientid
        self.__train_date = get_train_backdoor()
        self.__test_data = get_test_backdoor()
        # self.epsilon = 2  # 差分隐私预算
        # self.delta = 0.00001
        # self.sigma = np.sqrt(2 * (np.log(1.25 / self.delta))) / self.epsilon

    # 训练
    def train(self, globalmodel = None, epoch=1,):

        if globalmodel is not None:
            parms = self.model_my.state_dict().copy()
            for key, newvalue in zip(parms.keys(), globalmodel):
                parms[key] = torch.from_numpy(newvalue)
            self.model_my.load_state_dict(parms)
            #self.model_my.to(device)
        else:
            print("can not find global model")
            exit(0)

        for i in range(epoch):

            __dataloader_train_backdoor = DataLoader(self.__train_date,batch_size=128,shuffle=True)
            total_acc = 0  # 每一个round的准确率
            num = 0
            for data in __dataloader_train_backdoor:
                imgs, target = data
                imgs = imgs.to(device)
                target = target.to(device)
                output = self.model_my(imgs) + 1e-9
                self.optimizer.zero_grad()
                loss = self.loss_fnc(output, target)
                loss.backward()
                self.optimizer.step()
                pre_output = output.argmax(1)
                accuracy = (pre_output == target).sum()
                total_acc = total_acc + accuracy
                num += 1
                if num > 160:
                    break



            print(f"client {self.id} train accuracy is {total_acc / (num * self.batchsize)}")

        updateparm = []
        par = self.model_my.state_dict().copy()
        for key in par.keys():
            updateparm.append(par[key].cpu().numpy())
        updateparm = np.array(updateparm,dtype=object)
        grad =  globalmodel - updateparm

        return grad * 1.3  #放大自己梯度的影响，系数自己选。过大更容易被发现



    def myTest(self, model):

        tesrmodel = model

        test_data_size = len( self.__test_data)
        __dataloader_test_backdoor = DataLoader( self.__test_data,batch_size=128)
        tol_acc = 0
        tesrmodel.eval()
        for data in __dataloader_test_backdoor:
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)
            output = tesrmodel(imgs)
            pre_label = output.argmax(1)
            acc = (pre_label == label).sum()
            tol_acc += acc
        # torch.save(model, "model_test")
        print(f"client{self.id} backdoor attack accuracy is{tol_acc / test_data_size}")
        return tol_acc / test_data_size



if __name__ == '__main__':

    m = CovModel()
    model = []
    par = m.state_dict().copy()
    for key in par.keys():
        model.append(par[key].cpu().numpy())
    model = np.array(model,dtype=object)
    c = backdoorClient(initmodel=model)
    grad = c.train(globalmodel=model,epoch=20)
    # print()
    newmodel = model + grad
    c.myTest(c.model_my)



#






