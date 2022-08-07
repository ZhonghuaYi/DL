# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-08-07

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from python.dataset import MnistData, FashionMnistData
import python.trainmethod as trainmethod


def reshape(dataset):
    # 读取出的Mnist的特征是一维的，需要将其化为（通道，x，y）的形式
    dataset.features = dataset.features.view(-1, 1, 28, 28)
    return dataset


class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, paddings):
        super(NiNBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, strides, paddings),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)


class NiNNet(nn.Module):
    """NiN模型使用的数据集是ImageNet，其每张图片数据为3x224x224"""
    def __init__(self):
        super(NiNNet, self).__init__()
        self.conv = nn.Sequential(NiNBlock(3, 96, 11, 4, 0),
                                  nn.MaxPool2d(kernel_size=3, stride=2),
                                  NiNBlock(96, 256, 5, 1, 2),
                                  nn.MaxPool2d(kernel_size=3, stride=2),
                                  NiNBlock(256, 384, 3, 1, 1),
                                  nn.MaxPool2d(kernel_size=3, stride=2),
                                  nn.Dropout(0.5),
                                  NiNBlock(384, 10, 3, 1, 1),
                                  nn.AdaptiveAvgPool2d((1, 1)))
        self.flat = nn.Flatten()

    def forward(self, x):
        y = self.conv(x)
        return self.flat(y)


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224), device="cuda")
    net = NiNNet()
    net.to("cuda")
    print(f"Input tensor's shape:{x.shape}\n")
    print(net)
    y = net(x)
    print(f"Output tensor's shape:{y.shape}")
