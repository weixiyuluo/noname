# -*- codeing = utf-8 -*-
# @Time : 2024/2/21 21:53
# @Author : 李国锋
# @File: Server.py
# @Softerware:

import os
import random
from functools import reduce

import torch
import numpy as np
import torch.nn.functional as F
from Model import *
from client import *


seed = 3047
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Rserver:
    def __init__(self):
        self.aggregation = None  #聚合的梯度
        self.grad = []
        self.lr = 0.1  #学习率,其中客户端提交的梯度已经是grad*lr了，因此self.lr=1  ##如果担心客户端使用的lr过大，可以裁剪
        self.cilents = []  #在线客户端集合
        self.search = {}  #根据id搜索客户端的实列,数据可以删除
        self.dictionary = {} #根据id返回客户端加入的轮数，数据不应被删除
        self.G={}    #恢复时存储每个车辆的梯度向量
        self.weight_vector = None  #恢复时的模型向量
        self.globalmodel = CovModel().to(device)  # 全局模型
        self.get_model()
        self.optimizer = torch.optim.SGD(self.globalmodel.parameters(), lr=self.lr)#,momentum=0.9,weight_decay=0.1
        self.modellen = sum(p.numel() for p in self.globalmodel.parameters() )
        # self.blacklist = []   #投毒攻击车辆名单

    def get_model(self,model = None):
        if model is None:
            model = self.globalmodel
        self.model = []  # numpy数组形式的模型参数
        par = model.state_dict().copy()
        for key in par.keys():
            _ = par[key].cpu().numpy()
            self.model.append(_)
        self.model = np.array(self.model, dtype=object)


    def connectClient(self,id,round=0):
        if id in self.search.keys():
            print("id is exist")
            return None
        # elif id in self.blacklist:
        #     print("id in blacklisr")
        #     return None
        t = Client(clientid=id, initmodel=self.model)
        self.cilents.append(t)
        self.dictionary.update({id:round})  #{f"{id}":round}
        self.search.update({id:t})


    def exitFL(self,id):
        """
        客户端id退出FL，将其从self.search{}和self.cilents[]中删除
        """
        try:
            t = self.search.pop(id)
        except KeyError:
            print(f"client {id} is not exist")
            return None
        self.cilents.remove(t)
        print(f"client {id} is exit")
        return 1


    def vector_faltten(self,vector)->np:
        """
        vector 是一个不规则的二维numpy数组，(dim,),dim是第一维的维度，将其展成一维数组
        """
        t = np.zeros(0)
        for v in vector:
            v = v.reshape(-1)  # 展平
            t = np.concatenate([t, v])
        return t


    def detect(self):
        pass


    def savegrad(self,gradients,id,epoch):
        """
        保存的梯度是一维向量
        """
        gradient = self.vector_faltten(gradients)
        for i, value in enumerate(gradient):
            if value > 1e-5:           #这是client的学习率
                gradient[i] = 1
            elif value < 1e-5:
                gradient[i] = -1
            else:
                gradient[i] = 0
        g = gradient.astype(np.int8)
        np.save("./grad_history/cilent{}_grad_{}round".format(id, epoch), g)
        # np.save("./grad_history/cilent{}_poison_grad_{}round".format(id, epoch), gradient)
        # np.save("./grad_history/cilent{}_backdoor_grad_{}round".format(id, epoch), gradient)


    def setGrad(self):
        """
        用来将self.aggregation中的值赋给globalmodel，不能将保存的梯度的值赋给globalmodel
        """
        i = 0
        for param in self.globalmodel.parameters():  # self.model_my.model._modules.
            if param.requires_grad:
                param.grad=torch.from_numpy(self.aggregation[i].astype(np.float32)).type(torch.FloatTensor).cuda()
                i =i+1
        #torch.nn.utils.clip_grad_norm_(self.globalmodel.parameters(), max_norm=20)


    def setModelParam(self,model = None):
        """
        model是模型的numpy数组
        """
        if model is None:
            model = self.model

        par = self.globalmodel.state_dict().copy()
        for key, param in zip(par.keys(), model):
            par[key] = torch.from_numpy(param)
        self.globalmodel.load_state_dict(par, strict=True)


    def update(self, i=0):
        clientnum = len(self.cilents)
        if clientnum == 0:
            print("there is not client in FL")
            exit()
        print(f"epoch :{i+1}")
        self.get_model()
        self.optimizer.zero_grad()
        np.save(f"./model_history/model_{i}round",self.model)
        self.aggregation = []
        for j, vehice in enumerate(self.cilents):
            grad = vehice.train(globalmodel=self.model)   #接受梯度
            # self.detect(grad)
            self.savegrad(gradients=grad, id=vehice.id, epoch=i)
            self.aggregation.append(grad)
        sum_grad = reduce(lambda x, y: np.add(x, y), self.aggregation)
        self.aggregation = sum_grad/clientnum
        # print(self.aggregation)
        self.setGrad()
        self.optimizer.step()
        return self.search[1].myTest(self.globalmodel)
        # self.model -=self.aggregation


    def model_to_numpy(self,model=None):
        """
        # 将模型从tensor转为(n,1)numpy数组，默认转换globalmodel
        """
        if model is None:
            model = self.globalmodel
        params = torch.nn.utils.parameters_to_vector(model.parameters())  # 展成一维向量
        params = params.reshape(-1, 1)  ##变成[n,1]
        params = params.cpu().detach().numpy()  ##从GPU中放到cpu上，并转换成numpy数组
        return params


    def num2model(self,modelparam):  #      #matlab命名格式
        """
         numpy-->tensor-->model，将一维numpy数组转成模型
        """
        flat_model = np.squeeze(modelparam)   #(n,1)->(n,) # 列转行，不能直接使用转置，否则(1,5134)
        flat_model = torch.from_numpy(flat_model).type(torch.FloatTensor).cuda()
        torch.nn.utils.vector_to_parameters(flat_model,self.globalmodel.parameters())


    def LBFGS(self,weight, grad, v):
        weight_T = np.transpose(weight)
        A = np.matmul(weight_T, grad)
        ##计算对角矩阵并保留原矩阵的shape
        D = np.diag(np.diag(A))
        L = np.tril(A)
        # 计算σ
        grad_one_one = grad[..., -1]  # 第1列
        weight_one = weight[..., -1]
        # print(grad_one_one)
        # print(weight_one)
        grad_one_one_T = np.transpose(grad_one_one)
        weight_one_T = np.transpose(weight_one)
        result_up = np.matmul(grad_one_one_T, weight_one)
        result_low = np.matmul(weight_one_T, weight_one)
        sigma = result_up / result_low  # shape is (1,1)
        sigma = np.squeeze(sigma)  # 变成一个常数，如果带axis = 0的话变成一个一维数组
        ##计算P
        L_T = np.transpose(L)
        left_up = np.hstack((-1 * D, L_T))
        left_down_right = sigma * np.matmul(weight_T, weight)
        left_down = np.hstack((L, left_down_right))
        left = np.vstack((left_up, left_down))
        # print(left)
        left_inv = np.linalg.inv(left)

        grad_T = np.transpose(grad)
        right_up = grad_T.dot(v)
        right_down = sigma * weight_T.dot(v)
        right = np.vstack((right_up, right_down))
        p = left_inv.dot(right)
        ##计算Hv
        temp = np.concatenate((grad, sigma * weight), axis=1)
        temp = np.matmul(temp, p)
        Hv = sigma * v - temp
        return Hv



    def eraseing(self, id,nowround):

        if self.exitFL(id) is None:
            print(f"error id, client_{id} not online, eraseing abnormally")
            return None
        # self.blacklist.append(id)
        atten_round = self.dictionary[id]

        if atten_round < 2:   #小于2时无法初始化向量对
            self.model = np.load(f"./model_history/model_{atten_round}round")
            print("can not normally recover")
            self.setModelParam()
            return None

        else:
            ###恢复初始化
            print("**************************************************")
            print("before recovery:")
            # self.search[1].myTest(self.globalmodel)
            weight_hat = np.load(f"./model_history/model_{atten_round}round.npy",allow_pickle=True)
            weight_hat = self.vector_faltten(weight_hat).reshape((-1,1))  #模型参数需要变成(n,1)

            new_model1 = self.vector_faltten(np.load(f"./model_history/model_{atten_round - 1}round.npy",allow_pickle=True)).reshape((-1,1))
            new_model2 = self.vector_faltten(np.load(f"./model_history/model_{atten_round - 2}round.npy",allow_pickle=True)).reshape((-1,1))

            weight_vector = np.concatenate((new_model2 - weight_hat, new_model1 - weight_hat), axis=1) #原论文就是用axis=1连接
            # print(weight_vector.shape)
            G = {}
            for index, vehicle in enumerate(self.cilents):
                id = vehicle.id
                grad_hat = np.load(f"./grad_history/cilent{id}_grad_{atten_round}round.npy",
                                   allow_pickle=True).reshape((-1, 1))
                grad_one = np.load("./grad_history/cilent{}_grad_{}round.npy".format(id, atten_round - 1),
                                   allow_pickle=True).reshape((-1, 1))
                grad_two = np.load("./grad_history/cilent{}_grad_{}round.npy".format(id, atten_round - 2),
                                   allow_pickle=True).reshape((-1, 1))


                grad_vector = np.concatenate((grad_two - grad_hat, grad_one - grad_hat), axis=1)
                # print(grad_vector)
                G.update({id: grad_vector})
            acc = np.zeros(nowround-atten_round + 1)

            ###开始恢复
            for i in range(nowround-atten_round + 1):
                round = i + atten_round + 1
                print (f" the {i + 1} round recovery start")
                if (i+1)%20==0:
                    acc[i] =self.update(-1)
                    weight_hat = self.model_to_numpy(self.globalmodel)
                    continue
                g = np.zeros((self.modellen,1))
                weight_orig = self.vector_faltten(np.load(f"./model_history/model_{round}round.npy",allow_pickle=True)).reshape((-1,1))
                v = weight_hat - weight_orig
                for id,grad_vector in G.items():
                    rec_grad = self.LBFGS(weight_vector, grad_vector, v)
                    org_grad = np.load(f"./grad_history/cilent{id}_grad_{round}round.npy", allow_pickle=True)
                    org_grad = org_grad.reshape((-1,1))
                    rev_grad = rec_grad + org_grad
                    # np.save(f"./grad_recover/cilent{id}_{i}round",rev_grad)
                    g +=rev_grad
                l = len(G)
                g /= l
                for k,data in enumerate(g):
                    if data[0]>l:
                        g[k][0] = l
                    elif data[0]<-l:
                        g[k][0]=-l
                weight_hat = weight_hat -  0.0001*g
                # np.save(f"./model_recover/recmodel_{i}round",weight_hat)
                self.num2model(weight_hat)
                acc[i] = self.search[1].myTest(self.globalmodel)

            self.model = self.num2model(weight_hat)
            np.save(f"./accuaracs/test__acc", acc)


    def testacc(self):
        model = torch.load("./model_history/model_57round")
        self.cilents[0].myTest(model)





if __name__ =='__main__':
    s = Rserver()

    # c = np.load("./model_history/model_6round.npy",allow_pickle=True)
    # v = s.vector_faltten(c)
    # # print(v.shape)
    # v = s.vector_faltten(c).reshape(-1,1)
    # # print(v.shape)
    # s.num2model(v)
    #
    # s.connectClient(1)
    # s.connectClient(2)
    # s.update(999)

    #
    s.connectClient(1)
    s.connectClient(2)
    s.connectClient(3)
    s.connectClient(4,2)
    acc = np.zeros(200)
    # for r in range(200):
    #     if r == 2:
    #         s.connectClient(4, 2)
    #     acc[r] = s.update(r)
    # np.save("./train_accuracy",acc)

    #
    #
    s.eraseing(id=4,nowround=190)
    # # s.exitFL(4)
    #
    # for i in range(5):
    #     s.newupdate(i)
        #s.update(i)


    # # s.testacc()





