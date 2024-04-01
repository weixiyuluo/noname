

import random
import time
import torch
import numpy as np
import torchvision
from torch import nn
import torch.nn.functional  as F
from torch.utils.data import DataLoader
from Model import *
from FlipMNISTdataset import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")




class poisonClient():

    def __init__(self, clientid=0,initmodel = None,lr=0.001):
        self.batchsize = 128
        self.__train_data = NewMNIST(poison=True)
        self.__test_data = NewMNIST(train=False, poison=True,all=True)
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
        # self.epsilon = 2  # 差分隐私预算
        # self.delta = 0.00001
        # self.sigma = np.sqrt(2 * (np.log(1.25 / self.delta))) / self.epsilon

    # 训练
    def train(self, globalmodel = None, epoch=1,):


        __train_loader = DataLoader(self.__train_data, batch_size=self.batchsize, shuffle=True, drop_last=True)

        if globalmodel is not None:
            parms = self.model_my.state_dict().copy()
            for key, newvalue in zip(parms.keys(), globalmodel):
                parms[key] = torch.from_numpy(newvalue)
            self.model_my.load_state_dict(parms)
        else:
            print("can not find global model")
            exit(0)

        for i in range(epoch):
            # model_len  = sum(p.numel() for p in self.model_my.parameters() if p.requires_grad)
            # global_grad = np.zeros(model_len)  # 5134是梯度向量展成一维后的大小
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
                self.optimizer.step()
                pre_output = output.argmax(1)
                accuracy = (pre_output == target).sum()
                total_acc = total_acc + accuracy
                if(num > 120):   #恶意攻击者想扩大自己对模型更新的影响，但是又不敢训练次数过多以防止超时
                    break
                num+=1

            print(f"client {self.id} train accuracy is {total_acc / (num*self.batchsize)}")
        updateparm = []
        par = self.model_my.state_dict().copy()
        for key in par.keys():
            updateparm.append(par[key].cpu().numpy())
        updateparm = np.array(updateparm,dtype=object)
        grad =  globalmodel - updateparm
        return grad    # return grad * 2 #扩大影响，但是有可能会被检测出来



    def myTest(self, model):
        tesrmodel = model
        __test_load = DataLoader(self.__test_data, batch_size=128)
        tol_acc = 0
        # index = self.__test_data.get_poision_index()
        # all_preds = []
        tesrmodel.eval()
        for data in __test_load:
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)
            output = tesrmodel(imgs)
            _, preds = torch.max(output, 1)
            acc = (preds == label).sum()
            tol_acc += acc
            # all_preds.extend(preds.cpu().numpy())
        # all_preds = np.array(all_preds)
        # poisoned_preds = all_preds[index]
        # attack_success_rate = np.mean((poisoned_preds == 0) | (poisoned_preds == 5) | (poisoned_preds == 7))
        attack_success_rate = tol_acc/len(self.__test_data)
        print(f"malicous client{self.id} attack accuracy is{attack_success_rate}")
        return attack_success_rate


if __name__ == '__main__':

    m = CovModel()
    model = []
    par = m.state_dict().copy()
    for key in par.keys():
        model.append(par[key].cpu().numpy())
    model = np.array(model, dtype=object)
    c = poisonClient(initmodel=model)
    for i in range(5):
        c.train()
    c.myTest(c.model_my)
#







