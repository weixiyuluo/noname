# -*- codeing = utf-8 -*-
# @Time : 2024/2/19 20:02
# @Author : 李国锋
# @File: backdoorMNISTset.py
# @Softerware:
#这个DatasetBD主要还是应用于下载dataset的时候没有进行totensor转换，而是直接将img变为numpy类型再进行投毒，
#最后完成totensor转换并且合成一个dataset，后面可以直接变为dataloader

#如果是在下载dataset的时候直接进行了totensor转化，则直接从dataset中读取数据然后对tensor类型进行修改就行了吗？？

import torch
import torchvision
from tqdm import tqdm
from torchvision import datasets,transforms
import numpy as np
from torch.utils.data import DataLoader,Dataset
import time
from PIL import Image
from torch import nn



tf = transforms.Compose([torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.1307,), (0.3081,))
                         ])

__dataset_train_MINST = torchvision.datasets.MNIST(root="../data/MNIST", train=True, download=True)
__dataset_test_MNIST = torchvision.datasets.MNIST(root="../data/MNIST", train=False, download=True)




__inject_portion1 = 0.7   #插入后门的概率
__target_label1 = 2        #target label
__trig_h1 = 3             # 触发器大小
__trig_w1 = 3
target_type1 = "all2one"    #攻击类型

#这里的dataset并没有进行tensor的转化 是img类型 size为(32,32)，转化为numpy类型也是(32,32)
#所以获取img的weight和height是[0]和[1]  修改的时候是直接转化为numpy再img[j,k] = 255

#当然也可以是在下载的时候直接进行tensor转化，然后在投毒的时候就自己变，最后的transform看情况加
class DatasetBD(Dataset):
    def __init__(self, full_dataset, inject_portion, target_label, trig_w, trig_h, target_type, transform=None,
                 mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addtrigger(full_dataset, target_label, inject_portion, mode, distance, trig_w,
                                       trig_h, target_type)
        self.device = device
        self.transform = transform

    #def __getitem__(self, item): 方法是直接使用DatasetBD(num)就是执行getitem中的代码
    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
       # ind = self.dataset[item][2]
        img = self.transform(img)

        #return img, label, ind
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addtrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, target_type):
        # print("Generating " + mode + "bad Imgs")
        #1.首先根据投毒率和dataset的长度获得投毒下标的列表
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        #2.创建一个空的列表命名为dataset_ 列表中存的是元组（图片，标签，最后一位不知道是啥）
        #使用dataset_.append(())往里面添加数据
        # dataset_就是最终的投毒完之后的dataset
        dataset_ = list()

        cnt = 0
        #3.用tqdm创建一个length（dataset）的进度条，i从0到length（dataset）-1
        for i in range(len(dataset)):
            data = dataset[i]
            #4.判断投毒的方式，all2one还是all2all
            #all2one就是所有的目标标签都是一样的
            #all2all是目标标签不一样，一种方法是将目标标签设置为（原标签+1)%标签数
            if target_type == 'all2one':
                #4.判断模型
                #tarin：在投毒下标中就投毒
                #test：看下面
                # 训练数据集只要是投毒下标就投毒
                if mode == 'train':
                    #***下面是投毒方式，这里好像是img类型的，通过np.array(data[0])变为矩阵
                    #下面按个selectTrigger的思路就是修改指定区域的值（通过两个for循环），也就是修改像素值都变为255或者0啥的
                    img = np.array(data[0])
                    width = img.shape[0]
                    #print(width)  输出3
                    height = img.shape[1]
                    #print(height)  输出32 所以这不应该是width是1 height是1吗

                    if i in perm:
                        # select trigger
                        for j in range(width - distance - trig_w, width - distance):
                            for k in range(height - distance - trig_h, height - distance):
                                img[j, k] = 255.0
                                #img = self.selectTrigger(img, width, height, distance, trig_w, trig_h,
                                #                         trigger_type)

                                 # change target
                                 #这里重要，dataset_.append((img, target_label, 1))那个1不知道有啥用
                                 #dataset_.append((img, target_label, 1))
                        dataset_.append((img, target_label))
                        cnt += 1
                    #如果不在下标里面就直接添加到dataset_中
                    else:
                        dataset_.append((img, data[1]))
                        # dataset_.append((img, data[1], 0))

                    # 这个else是测试数据集
                    # else下的第一个if的意思是
                    # 如果注射率不等于0，则只在原标签不为目标标签的数据中选择在perm的下标投毒
                    #原标签为目标标签的数据不存入到dataset_中吗？？
                    # 如果注射率等于0，则perm为空，则不投毒，全部直接添加到dataset_中
                else:
                            if data[1] == target_label and inject_portion != 0.:
                                    continue

                            img = np.array(data[0], dtype=np.uint8)
                            width = img.shape[0]
                            height = img.shape[1]
                            if i in perm:
                                    # self.selectTrigger投毒 这里好像就是直接将原图片的一部分改变像素值
                                    for j in range(width - distance - trig_w, width - distance):
                                        for k in range(height - distance - trig_h, height - distance):
                                            img[j, k] = 255.0

                                    dataset_.append((img, target_label))
                                    #dataset_.append((img, target_label, 0))
                                    cnt += 1
                            else:
                                    dataset_.append((img, data[1]))
                                    #dataset_.append((img, data[1], 1))

            # all2all attack
            # 这里训练和测试的投毒数据选择一样？？
            elif target_type == 'all2all':

                if mode == 'train':
                            img = np.array(data[0])
                            width = img.shape[0]
                            height = img.shape[1]
                            if i in perm:

                                for j in range(width - distance - trig_w, width - distance):
                                    for k in range(height - distance - trig_h, height - distance):
                                        img[j, k] = 255.0
                                    # self._change_label_next：label_new = ((label + 1) % 10)
                                    target_ = (data[1] + 1) % 10

                                    dataset_.append((img, target_))
                                    cnt += 1
                            else:
                                    dataset_.append((img, data[1]))

                else:

                            img = np.array(data[0])
                            width = img.shape[0]
                            height = img.shape[1]
                            if i in perm:
                                for j in range(width - distance - trig_w, width - distance):
                                    for k in range(height - distance - trig_h, height - distance):
                                        img[j, k] = 255.0

                                    target_ = (data[1] + 1) % 10
                                    dataset_.append((img, target_))
                                    cnt += 1
                            else:
                                    dataset_.append((img, data[1]))

            # clean label attack
            #干净标签攻击：
            elif target_type == 'cleanLabel':

                    if mode == 'train':
                            img = np.array(data[0], dtype=np.uint8)
                            width = img.shape[0]
                            height = img.shape[1]

                            if i in perm:
                                    #在原标签为目标标签的数据中攻击
                                    #但是这样再加一个判断的话投毒率不就不对了吗，真实投毒的数据变少
                                    if data[1] == target_label:

                                        for j in range(width - distance - trig_w, width - distance):
                                            for k in range(height - distance - trig_h, height - distance):
                                                img[j, k] = 255.0

                                            dataset_.append((img, data[1]))
                                            cnt += 1

                                    else:
                                            dataset_.append((img, data[1]))
                            else:
                                    dataset_.append((img, data[1]))

                    else:
                        #测试的时候只在原标签不是目标标签的数据中投毒
                        #原标签为目标标签的数据都不加入到dataset_中吗？？
                        if data[1] == target_label:
                            continue

                        img = np.array(data[0], dtype=np.uint8)
                        width = img.shape[0]
                        height = img.shape[1]
                        #不是目标标签且在投毒下标里的
                        if i in perm:
                            for j in range(width - distance - trig_w, width - distance):
                                for k in range(height - distance - trig_h, height - distance):
                                    img[j, k] = 255.0

                            dataset_.append((img, target_label))
                            cnt += 1
                        else:
                            dataset_.append((img, data[1]))

        time.sleep(0.01)
        #这里我感觉应该修改一下，因为
        # print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_

#这里的DatasetBD的输入的数据集需要是img类型的，和投毒的方式有关，因为这里是直接将img转化为numpy再修改对应位置的像素值

def get_train_backdoor():
    return  DatasetBD(full_dataset=__dataset_train_MINST, inject_portion=__inject_portion1, target_label=__target_label1, trig_w=__trig_w1, trig_h=__trig_h1, target_type=target_type1, transform=tf, mode="train")

def get_test_backdoor():
    return  DatasetBD(full_dataset=__dataset_test_MNIST, inject_portion=1, target_label=__target_label1, trig_w=__trig_w1, trig_h=__trig_h1, target_type=target_type1, transform=tf, mode="test")

# dataset_test_clean = DatasetBD(full_dataset=__dataset_test_MNIST, inject_portion=0, target_label=__target_label1, trig_w=__trig_w1, trig_h=__trig_h1, target_type=target_type1, transform=tf, mode="test")


# dataloader_train_backdoor = DataLoader(dataset_train_backdoor,batch_size=64)
# dataloader_test_backdoor = DataLoader(dataset_test_backdoor,batch_size=64)
# dataloader_test_clean = DataLoader(dataset_test_clean,batch_size=64)




# if __name__=='__main__':
#     class BadNet(nn.Module):
#         def __init__(self, num_classes):
#             super(BadNet,self).__init__()
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(1,16,kernel_size=5,padding=2),
#                 nn.BatchNorm2d(16),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2)
#             )
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(16, 32, kernel_size=5, padding=2),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2)
#             )
#             self.conv3 = nn.Sequential(
#                 nn.Conv2d(32, 64, kernel_size=5, padding=2),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2)
#             )
#             self.linear=nn.Linear(in_features=3*3*64,out_features=num_classes)
#
#         def forward(self,x):
#             out=self.conv1(x)
#             out = self.conv2(out)
#             out=self.conv3(out)
#             out = out.view(out.size(0),-1)
#             out=self.linear(out)
#             return out
#
#
#     #实例化模型
#
#     badNet = BadNet(10)
#     #如果有训练好的模型参数存放在"xxx"
#     #则可以用badNet.load_state_dict(torch.load("xxx"))
#
#     #训练和测试
#     #训练和测试是首先定义loss 优化器 参数（epoch,total_step）,然后在每一轮epoch中训练一次train，分别测试一次投毒测试集和干净测试集
#
#
#     #首先定义loss 优化器 一些指标参数
#     loss1 = torch.nn.CrossEntropyLoss()
#     lr1 = 0.0005
#     optimizer = torch.optim.SGD(badNet.parameters(), lr=lr1)
#
#     epoch = 100
#     total_train_step = 0
#     total_test_step = 0
#     for i in range(epoch):
#
#         #训练
#         print(f"-------第{i + 1}轮训练开始-------")
#         badNet.train()
#         total_loss_train = 0
#         for data in dataloader_train_backdoor:
#             imgs, targets = data
#             # print(imgs.shape)
#             outputs = badNet(imgs)
#             loss = loss1(outputs, targets)
#             total_loss_train += loss
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_train_step += 1
#             if total_train_step % 100 == 0:
#                 print(f"完成{total_train_step}次训练时，loss为:{loss}")
#
#
#         #total_train_step += 1
#         #avg_loss = total_loss_train / len(dataset_test_backdoor)
#
#         #if total_train_step % 10 == 0:
#         #    print(f"第{total_train_step}轮训练的平均loss为：{avg_loss}")
#
#
#         #测试
#         total_acc = 0
#         with torch.no_grad():
#             for data in dataloader_test_backdoor:
#                 imgs, targets = data
#                 outputs = badNet(imgs)
#                 acc = (outputs.argmax(1) == targets).sum()
#                 total_acc += acc
#
#         total_test_step += 1
#         print(f"第{total_test_step}轮训练时的投毒测试集的准确个数为{total_acc}")
#         asr = total_acc / len(__dataset_test_MNIST.targets)
#         print(f"第{total_test_step}轮训练时的asr为:{asr}")
#
#         total_acc = 0
#         with torch.no_grad():
#             for data in dataloader_test_clean:
#                 imgs,targets = data
#                 outputs = badNet(imgs)
#                 acc = (outputs.argmax(1) == targets).sum()
#                 total_acc += acc
#
#         print(f"第{total_test_step}轮训练时的干净测试集的准确个数为{total_acc}")
#         ba = total_acc / len(__dataset_test_MNIST.targets)
#         print(f"第{total_test_step}轮训练时的ba为{ba}")









