# -*- codeing = utf-8 -*-
# @Time : 2024/2/21 21:53
# @Author : 李国锋
# @File: Server.py
# @Softerware:

import os
import random
import time
from functools import reduce
import torch
import numpy as np
import torch.nn.functional as F
from Model import *
from OurSheme.client import Client
from flipMNISTClient import poisonClient
from backdoorMNISTClient import backdoorClient
from gtrsbClient import GTRSBClient
from miniClient import miniClient





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Just a simple sandbox for testing out python code, without using Go.
def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()


import signal
signal.signal(signal.SIGINT, debug_signal_handler)

class Rserver:
    def __init__(self,lr = 0.1,time_limit = 5*60*60):
        self.aggregation = None  #聚合的梯度
        self.grad = []
        self.lr = lr  #学习率,其中客户端提交的梯度已经是grad*lr了，因此self.lr=1  ##如果担心客户端使用的lr过大，可以裁剪
        self.cilents = []  #在线客户端集合
        self.search = {}  #根据id搜索客户端的实列,数据可以删除
        self.dictionary = {} #根据id返回客户端加入的轮数，数据不应被删除
        self.G={}    #恢复时存储每个车辆的梯度向量
        self.weight_vector = None  #恢复时的模型向量

        self.globalmodel = CovModel().to(device)  # 全局模型
        # self.globalmodel = GTRSBmodel().to(device)

        self.get_model()
        self.optimizer = torch.optim.SGD(self.globalmodel.parameters(), lr=self.lr)#,momentum=0.9,weight_decay=0.1
        self.model_len = sum(p.numel() for p in self.globalmodel.parameters())
        # self.blacklist = []   #车辆黑名单
        self.time_limit = time_limit

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
        """"
        根据不同的数据集/模型选择不同的客户端
        """
        if id in self.search.keys():
            print("id is exist")
            return None
        # elif id in self.blacklist:
        #     print("id in blacklisr")
        #     return None

        t = Client(clientid=id, initmodel=self.model)
        # t = miniClient(clientid=id, initmodel=self.model)
        # t = GTRSBClient(clientid=id, initmodel=self.model)

        self.cilents.append(t)
        self.dictionary.update({id:round})  #{f"{id}":round}
        self.search.update({id:t})

    def connectflipClient(self,id,round=0):
        """
        执行反转攻击
        """
        if id in self.search.keys():
            print("id is exist")
            return None
        t = poisonClient(clientid=id, initmodel=self.model)
        self.cilents.append(t)
        self.dictionary.update({id:round})  #{f"{id}":round}
        self.search.update({id:t})

    def connectBackdoorClient(self,id,round=0):
        """
        执行后门攻击
        """
        if id in self.search.keys():
            print("id is exist")
            return None
        t = backdoorClient(clientid=id, initmodel=self.model)
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
        vector 是一个不规则的二维numpy数组，将其形状变成(dim,)也就是一维数组
        """
        t = np.zeros(0)
        for v in vector:
            v = v.reshape(-1)  # 展平
            t = np.concatenate([t, v])
        return t


    def detect(self):
        """
        this is a poison attack methods. our scheme is suit for Most poisoning attack detection algorithms.
        because of the limit of time, we do not finish it.
        """
        pass

    # 将数字编码为2位的形式
    def encode_number(self,num):
        return num & 0b11      #0b表示2进制

    # 将四个数字编码并存储到一个np.int8中，每个数字都是0，1，或者-1。这样每个梯度元素只占2bit
    def encode_to_byte(self,n1, n2, n3, n4):
        """
        将4个int8类型的数字（数字为-1，1，0）放到一个np.int8变量中进行保存
        """
        byte = (self.encode_number(n1) << 6) | (self.encode_number(n2) << 4) | (
                    self.encode_number(n3) << 2) | self.encode_number(n4)
        return np.int8(byte)

    # 从np.int8中提取四个数字，每个数字都是0，1，或者-1
    def decode_from_byte(self,byte):
        """
        从一个np.int8类型的变量中取出4个模型梯度方向元素
        """
        nums = []
        for i in range(3, -1, -1):
            num = (byte >> (i * 2)) & 0b11
            # 将2位的数字解码回原来的数字
            if num == 0b11:  # 解码 -1
                num = -1
            nums.append(num)
        return nums

    def savegrad(self,gradients,id,epoch,flag =False):
        """
        保存的梯度是一维向量,flag代表是否保存的时候将4个element放到一个地址中
        flag = True 的加载代码因时间关系并未完成，我们会在之后进行补充
        """
        gradient = self.vector_faltten(gradients)
        # np.save("./grad_history_all",gradient)
        for i, value in enumerate(gradient):
            if value > 1e-7:          #减小误差
                gradient[i] = 1
            elif value < -1e-7:
                gradient[i] = -1
            else:
                gradient[i] = 0
        g = gradient.astype(np.int8)          #每个元素只占2bit
        self.is_2bit = flag                #是否使用2bit进行存储
        if(self.is_2bit):                 #np.int8  占8bit，因此我们可以将4个元素放在一个int8中
            g_2 = np.zeros(0)
            for i in range(0,len(g),4):       # 将四个数字编码并存储到一个np.int8中，每个数字都是0，1，或者-1。这样每个梯度元素只占2bit
                _ = self.encode_to_byte(g[i],g[i+1],g[i+2],g[i+3])  #每4个数字放一个np.int8中
                g_2 = np.append(g_2,_)

        np.save("./grad_history/cilent{}_grad_{}round".format(id, epoch), g)
        # np.save("F:/iov/grad_history/cilent{}_grad_{}round".format(id, epoch), g_2)
        # np.save("./grad_history/cilent{}_poison_grad_{}round".format(id, epoch), gradient)
        # np.save("./grad_history/cilent{}_backdoor_grad_{}round".format(id, epoch), gradient)


    def setGrad(self):
        """
        用来将self.aggregation中的值赋给globalmodel.grad以使用优化器来更新全局模型
        """
        i = 0
        for param in self.globalmodel.parameters():  # self.model_my.model._modules.
            if param.requires_grad:
                param.grad=torch.from_numpy(self.aggregation[i].astype(np.float32)).type(torch.FloatTensor).cuda()
                i =i+1
        #torch.nn.utils.clip_grad_norm_(self.globalmodel.parameters(), max_norm=20)


    def setModelParam(self,model = None):
        """
        model是numpy数组类型的模型参数，将其加载到全局模型中
        """
        if model is None:
            model = self.model

        par = self.globalmodel.state_dict().copy()
        for key, param in zip(par.keys(), model):
            par[key] = torch.from_numpy(param)
        self.globalmodel.load_state_dict(par, strict=True)


    def update(self, i=0,modelp =None):
        """
        收集梯度，更新全局模型
        """
        clientnum = len(self.cilents)
        if clientnum == 0:
            print("there is not client in FL")
            exit()
        print(f"epoch :{i+1}")
        if modelp is not  None:
            self.setModelParam(modelp)
        self.get_model()
        self.optimizer.zero_grad()
        np.save(f"./model_history/model_{i}round",self.model)
        self.aggregation = []
        weight_all = 0
        num_all = 0
        T1 = time.time()
        random.shuffle(self.cilents)  #打乱，模拟部分车辆没办法即时上传
        for j, vehice in enumerate(self.cilents):
            T2 = time.time()
            # if(T2 - T1)>self.time_limit:   #到达一定的时间后中至收集，会消耗较大的cpu，linux可以通过 signal 模块来设置一个定时器，当时间到时，会触发一个异常，让你的程序捕获并处理。
            #     break                       # or if num_all > number_limit: break 。number_limit是自己设定的接受客户端的数量，例如0.7*clientnum
            grad = vehice.train(globalmodel=self.model)   #接受梯度
            # grad,w = vehice.train(globalmodel=self.model)
            # self.detect(grad)
            self.savegrad(gradients=grad, id=vehice.id, epoch=i)  ## related code of flag=True does not finish because of the limit of time, we will finish it later
            self.aggregation.append(grad)
            # self.aggregation.append(grad*w)
            # weight_all +=w
            num_all +=1
        sum_grad = reduce(lambda x, y: np.add(x, y), self.aggregation)
        self.aggregation = sum_grad/num_all
        # self.aggregation = sum_grad / (num_all*weight_all)
        # print(self.aggregation)
        self.setGrad()
        self.optimizer.step()
        # return self.search[1].myTest(self.globalmodel)


    def model_to_numpy(self,model=None):
        """
        # 将模型从tensor转为(n,1)形状的numpy数组，默认使用globalmodel转换
        """
        if model is None:
            model = self.globalmodel
        params = torch.nn.utils.parameters_to_vector(model.parameters())  # 展成一维向量
        params = params.reshape(-1, 1)  ##变成[n,1]
        params = params.cpu().detach().numpy()  ##从GPU中放到cpu上，并转换成numpy数组
        return params


    def num2model(self,modelparam):  #      #matlab命名格式
        """
         (n,1)numpy-->tensor-->model，将输入形状是（n,1）的numpy数组放到模型参数中
        """
        flat_model = np.squeeze(modelparam)   #(n,1)->(n,) # 列转行，不能直接使用转置，否则(1,n)
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
        try:
            left_inv = np.linalg.inv(left)
        except np.linalg.LinAlgError:    #求不了逆， left_inv是奇异矩阵
            return 0
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


    def eraseing(self, id,nowround,lr = 0.0001,threshold = 10,get_grad_round = 100, updates_round = 13,get_round =1):
        """
        lr是恢复时候的学习率，threshold是梯度裁剪阈值，get_grad_round是多久获得一次真实梯度
        updates_round 是多久更新一次vector pairs（由于篇幅限制论文中没写）， get_round是每次获得真实梯度时需要获得几轮真实梯度
        注意阈值是客户端的梯度*客户端自己的学习率，因此比较小.
        当更新准确率提升较小的时候增加学习率.
        当准确率稳步下降的时候更新vector pairs.
        使用2bit进行存储的话需要使用self.decode_from_byte()  。时间关系并未完成
        此外，构建vector pairs 时我们默认车辆全部在线。因为如果不在线的话构建vector pairs的代码比较复杂
        """
        T1 = time.time()
        ##self.decode_from_byte()   # related code does not finish
        if self.exitFL(id) is None:
            print(f"error id, client_{id} not online, eraseing abnormally")
            return None
        # self.blacklist.append(id)
        atten_round = self.dictionary[id]
        if atten_round < 2:   #小于2时无法初始化向量对
            self.model = np.load(f".filename/model_{atten_round}round")
            print("can not normally recover， you need to retrain")
            self.setModelParam()
            return None
        else:
            ###恢复初始化
            print("**************************************************")

            # maybe we need use this two lines codes to decide how to build vector pairs, however , is too difficult so i will finish it later
            # # 指定要读取的文件夹路径
            # folder_path = "./model_history"            #
            # # 获取文件夹内所有文件的列表
            # file_list = os.listdir(folder_path)

            ## 构建 vector pairs
            weight_hat = np.load(f"./model_history/model_{atten_round}round.npy",allow_pickle=True)
            weight_hat = self.vector_faltten(weight_hat).reshape((-1,1))  #模型参数需要变成(n,1)
            new_model1 = self.vector_faltten(np.load(f"./model_history/model_{atten_round - 1}round.npy",allow_pickle=True)).reshape((-1,1))
            new_model2 = self.vector_faltten(np.load(f"./model_history/model_{atten_round - 2}round.npy",allow_pickle=True)).reshape((-1,1))
            weight_vector = np.concatenate((new_model2 - weight_hat, new_model1 - weight_hat), axis=1) #原论文就是用axis=1连接
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
                G.update({id: grad_vector})
            acc = np.zeros(nowround-atten_round + 1)
            ###开始恢复
            for i in range(nowround-atten_round):
                round = i + atten_round + 1
                print (f" the {i + 1} round recovery start")

                # if (i + 1) % get_grad_round == 0:
                #     for pp in range(get_round):
                #         acc[i] =self.update(-1)
                #     weight_hat = self.model_to_numpy(self.globalmodel)
                #     continue

                g = np.zeros((self.model_len, 1))
                weight_orig = self.vector_faltten(np.load(f"./model_history/model_{round}round.npy",allow_pickle=True)).reshape((-1,1))
                v = weight_hat - weight_orig
                rev_grad = {}
                for id,grad_vector in G.items():
                    rec_grad = self.LBFGS(weight_vector, grad_vector, v)
                    org_grad = np.load(f"./grad_history/cilent{id}_grad_{round}round.npy", allow_pickle=True)
                    org_grad = org_grad.reshape((-1,1))
                    _= rec_grad + org_grad
                    _ = np.clip(_, a_min = -threshold*0.001, a_max =threshold*0.001)
                    rev_grad.update({id:_})
                    # rev_grad *= w[id]   #w是服务器记录的客户端的数据集大小
                    g +=rev_grad[id]
                l = len(G)
                g /= l
                # g/= weight_all   # weight += w for w in w[i]  #weight_all是客户端数据集大小的总和

                # g = np.clip(g, a_min = -threshold, a_max =threshold)  #阈值自己设定
                weight_hat = weight_hat - lr * g      #与训练相同的学习率，或者再小一点的lr
                self.num2model(weight_hat)

                # acc[i] = self.search[1].myTest(self.globalmodel)

                # self.testBackdoor()
                # self.get_model()
                # np.save(f"F:/iov/recover_gtrsb_model/rrecover_{i}round",self.model)  #保存恢复的模型

                # update vector pairs is need   #however, 有时候不更新反而效果更好
                if (i + 1) % updates_round == 0:
                    new_model2 = new_model1
                    new_model1 = weight_hat
                    weight_vector = np.concatenate((new_model2 - weight_orig, new_model1 - weight_orig),
                                                   axis=1)  # 原论文就是用axis=1连接
                    G = {}
                    for index, vehicle in enumerate(self.cilents):
                        id = vehicle.id
                        org_grad = np.load(f"./grad_history/cilent{id}_grad_{round}round.npy",
                                           allow_pickle=True).reshape((-1, 1))
                        grad_two = grad_one
                        grad_one = rev_grad[id]
                        grad_vector = np.concatenate((grad_two - org_grad, grad_one - org_grad), axis=1)
                        G.update({id: grad_vector})
                # 结束更新
            self.num2model(weight_hat)
            self.get_model()
            T2 = time.time()

            self.search[1].myTest(self.globalmodel)
            print("running time is",T2-T1)
            np.save("./accuaracs/minist_new", acc)

    def testacc(self):
        model = np.load("./model_history/model_100round.npy",allow_pickle=True)
        self.cilents[0].myTest(model,1)

    def testFilp(self):
        t = poisonClient(clientid=0, initmodel=self.model)
        return t.myTest(self.globalmodel)


    def testBackdoor(self):
        t = backdoorClient(initmodel=self.model)
        return t.myTest(self.globalmodel)


if __name__ =='__main__':
    # import keyboard   #测试用
    s = Rserver()
    for i in range(1,5):
        s.connectClient(i)

    #
    s.connectClient(5,2)
    # acc = np.zeros(101)

    # T1 = time.time()
    # for r in range(100):
    #     s.update(r)
    #     if r == 97:
    #         T2 = time.time()
    #
    #
    # #     acc[r] = s.update(r)
    # #     if r == 2:
    # #         s.connectClient(5,r)
    # #         # s.connectBackdoorClient(5, r)
    #
    # #     # if keyboard.is_pressed('q'):
    # #     #     print("检测到'q'键，即将退出循环。")
    # #     #     break
    # # np.save("./minist_acc",acc)
    #
    # print("train time is:", T2 - T1)

    #阈值threshold = 0.001（客户端的学习率）*100（客户端数量）*预期为客户端设定的阈值（1） = 0.1
    # NOTICE! update the vector pairs also lead to accuracy do
    # 此外，以部分数字（例如50）当作更新向量对的周期会导致之后的模型准确率下降
    s.eraseing(id=5,nowround=100,lr=0.1*0.001,threshold=10,updates_round=21)  #学习率最好和训练时相同,或者更小  #先将放大的缩小








