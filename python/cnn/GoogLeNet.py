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
        self.conv4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding=1),
                                   nn.Conv2d(in_channels, c4, kernel_size=1),
                                   nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)


if __name__ == '__main__':
    pass
