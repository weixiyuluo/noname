# -*- codeing = utf-8 -*-
# @Time : 2024/2/19 10:09
# @Author : 李国锋
# @File: Model.py
# @Softerware:
import torch
from torch import nn


class CovModel(nn.Module):
    def __init__(self):
        super(CovModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=9,stride=1),
            # nn.BatchNorm2d(10),    #正则化层有均值和方差，恢复的时候无法恢复
            nn.ReLU(),
            nn.Conv2d(in_channels=10,out_channels=20,kernel_size=9,stride=1),
            nn.Flatten(),
            nn.Linear(2880,512),
            # nn.LayerNorm(512),
            nn.Linear(512,10)

        )

    def forward(self,input):
        return self.model(input)
        # x = self.model(input)
        #print(x.shape)                         #[64,10]  第一维是batchsize，最后一维是类别
        # return nn.functional.softmax(x,dim=-1)   #client所使用的损失函数中已经包含softmax()，不能再设置一遍



if __name__ == '__main__':
    item = CovModel()
    input = torch.ones(64,1,28,28)
    output = item(input)