# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-08-11

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


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        if in_channels == out_channels:
            self.conv2 = None
        else:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        if self.conv2:
            y = y + self.conv2(x)
        else:
            y = y + x
        y = self.relu(y)
        return y


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(Residual(64, 64),
                                Residual(64, 64))
        self.b3 = nn.Sequential(Residual(64, 128),
                                Residual(128, 128))
        self.b4 = nn.Sequential(Residual(128, 256),
                                Residual(256, 256))
        self.b5 = nn.Sequential(Residual(256, 512),
                                Residual(512, 512))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        y = self.b1(x)
        y = self.b2(y)
        y = self.b3(y)
        y = self.b4(y)
        y = self.b5(y)
        y = self.pool(y)
        y = self.flat(y)
        y = self.linear(y)
        return y


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224), device="cuda")
    net = ResNet18()
    net.to("cuda")
    print(f"Input tensor's shape:{x.shape}\n")
    print(net)
    y = net(x)
    print(f"Output tensor's shape:{y.shape}")
    print("Number of net's parameters:", sum(param.numel() for param in net.parameters()))
