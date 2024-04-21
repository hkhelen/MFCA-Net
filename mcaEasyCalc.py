# -*- coding:utf-8 -*-
# @Time       :2022/9/8 上午11:07
# @AUTHOR     :DingKexin
# @FileName   :CALC.py
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
# from HWDdownsampling import Down_wt
from CondensedAttention import CondensedAttentionNeuralBlock
from ca_module import CALayer
from mca_module import MCALayer
from eca_module import ECALayer

class Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=l1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)  # add pool
            # Down_wt(32,32) # 尝试将最大池化更改为haar小波下采样，观察效果
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=l2,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # add pool
            # Down_wt(32,32)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(2),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
            # Down_wt(64,64)
            # nn.Dropout(0.5),

        )
        # self.conv2_2 = nn.Sequential(
        #     nn.Conv2d(32, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     # nn.MaxPool2d(2),
        #     nn.ReLU(),  # No effect on order
        #     nn.MaxPool2d(2),
        #     # Down_wt(64,64)
        #     # nn.Dropout(0.5),

        # )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # add pool
            # Down_wt(128,128) 
            # nn.Dropout(0.5),


        )
        
        # self.conv4_1 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2), # add pool
        #     # Down_wt(128,128) 
        #     # nn.Dropout(0.5),
        # )
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

        # self.attention1=CondensedAttentionNeuralBlock(32)
        # self.attention2=CondensedAttentionNeuralBlock(64,[2,2],2)
        # CA
        # self.attention1=CALayer(32)
        # self.attention2=CALayer(64)
        # self.attention3=CALayer(128)

        self.attention1=MCALayer(32)
        self.attention2=MCALayer(64)
        self.attention3=MCALayer(128)
        self.attention4=MCALayer(256)

    def forward(self, x1, x2):
        # x1:16*64*12*12
        # x2:16*2*12*12
        x1_1 = self.conv1_1(x1)  
        x1_2 = self.conv1_2(x2)  
        x1_add = x1_1 * self.xishu1 + x1_2 * self.xishu2 # 16*32*6*6
        x1_add = self.attention2(x1_add)
        x2_1 = self.conv2_1(x1_1)  
        x2_2 = self.conv2_1(x1_2)  
        x2_add = x2_1 * self.xishu1 + x2_2 * self.xishu2 # 16*64*3*3
        x2_add = self.attention3(x2_add)
        x3_1 = self.conv3_1(x2_1)  
        x3_2 = self.conv3_1(x2_2)  
        x3_add = x3_1 * self.xishu1 + x3_2 * self.xishu2 # 16*128*1*1
        x3_add = self.attention4(x3_add)
        return x1_add, x2_add, x3_add



class Classifier(nn.Module):
    def __init__(self, Classes):
        super(Classifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),
        )
 
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.31]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.33]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([0.36]))

        # self.attention1=CALayer(64)
        # self.attention2=CALayer(32)
        # self.attention3=CALayer(16)
        # self.attention4=CALayer(11)
        self.attention4=MCALayer(128)
        self.attention1=MCALayer(64)
        self.attention2=MCALayer(32)
        self.attention3=MCALayer(16)

    def forward(self, x1, x2, x3):
        x1_1 = self.conv1(x1)  # 16*128*1*1 -> 16*64*1*1
        x1_1 = self.attention4(x1_1)
        x1_2 = self.conv2(x1_1)  # 16*64*1*1 -> 16*32*1*1
        x1_2 = self.attention2(x1_2)
        x1_3 = self.conv3(x1_2)  # 16*32*1*1 -> 16*11*1*1
        # x1_3 = self.attention4(x1_3)
        x1_3 = x1_3.view(x1_3.size(0), -1)  # 64*15

        x2_1 = self.conv1_2(x2)  # 16*64*3*3 -> 16*32*3*3
        x2_1 = self.attention2(x2_1)
        x2_2 = self.conv2_2(x2_1)  # 16*32*3*3 -> 16*16*1*1
        # x2_2 = self.attention2(x2_2)
        x2_3 = self.conv3_2(x2_2)  # 16*16*1*1 -> 16*11*1*1
        # x2_3 = self.attention4(x2_3)
        x2_3 = x2_3.view(x2_3.size(0), -1)  # 64*15

        x3_1 = self.conv1_3(x3)  # 16*32*6*6 -> 16*16*6*6
        x3_1 = self.attention2(x3_1)
        x3_2 = self.conv2_3(x3_1)  # 16*16*6*6 -> 16*16*1*1
        # x3_2 = self.attention3(x3_2)
        x3_3 = self.conv3_3(x3_2)  # 16*16*1*1 -> 16*11*1*1
        # x3_3 = self.attention4(x3_3)
        x3_3 = x3_3.view(x3_3.size(0), -1)  # 64*15
        x = x1_3+ x2_3 + x3_3

        return x
class Network(nn.Module):
    def __init__(self, l1, l2, Classes):
        super(Network, self).__init__()
        self.encoder = Encoder(l1=l1, l2=l2)
        self.classifier = Classifier(Classes=Classes)

    def forward(self, x1, x2):
        ex1, ex2, ex3 = self.encoder(x1, x2)
        output = self.classifier(ex3, ex2, ex1)
        return output

if __name__== "__main__":
    net = Network(l1=1, l2=1, Classes=11)
    x1=torch.randn(256,1,16,16)
    x2=torch.randn(256,1,16,16)
    output=net(x1,x2)
    print(output.shape)





