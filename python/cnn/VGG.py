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


class VGGBlock(nn.Module):
    def __init__(self, num, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.block = []
        for i in range(num):
            self.block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.block.append(nn.ReLU())
            in_channels = out_channels
        self.block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg = nn.Sequential(*self.block)

    def forward(self, x):
        return self.vgg(x)


class VGG11(nn.Module):
    """VGG11使用的数据集是ImageNet，其每张图片数据为3x224x224"""
    def __init__(self):
        super(VGG11, self).__init__()
        self.vgg11 = nn.Sequential(VGGBlock(1, 3, 64),
                                   VGGBlock(1, 64, 128),
                                   VGGBlock(2, 128, 256),
                                   VGGBlock(2, 256, 512),
                                   VGGBlock(2, 512, 512))
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(),
                                    nn.Linear(4096, 4096), nn.ReLU(),
                                    nn.Linear(4096, 10))

    def forward(self, x):
        y = self.vgg11(x)
        y = self.flat(y)
        y = self.linear(y)
        return y


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224), device="cuda")
    net = VGG11()
    net.to("cuda")
    print(f"Input tensor's shape:{x.shape}\n")
    print(net)
    y = net(x)
    print(f"Output tensor's shape:{y.shape}")
