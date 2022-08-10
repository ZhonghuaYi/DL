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


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, c2[0], kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, c3[0], kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(in_channels, c4, kernel_size=1),
                                   nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(64, 64, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block1 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                    Inception(256, 128, (128, 192), (32, 96), 64),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                    Inception(512, 160, (112, 224), (24, 64), 64),
                                    Inception(512, 128, (128, 256), (24, 64), 64),
                                    Inception(512, 112, (144, 288), (32, 64), 64),
                                    Inception(528, 256, (160, 320), (32, 128), 128),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                    Inception(832, 384, (192, 384), (48, 128), 128),
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten())
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224), device="cuda")
    net = GoogLeNet()
    net.to("cuda")
    print(f"Input tensor's shape:{x.shape}\n")
    print(net)
    y = net(x)
    print(f"Output tensor's shape:{y.shape}")
