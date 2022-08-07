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


class AlexNet(nn.Module):
    """AlexNet使用的数据集是ImageNet，其每张图片数据为3x224x224"""
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2),
                                  nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2),
                                  nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                                  nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                                  nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2))
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(6400, 4096), nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096), nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 10))

    def forward(self, x):
        y = self.conv(x)
        y = self.flat(y)
        y = self.linear(y)
        return y


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224), device="cuda")
    net = AlexNet()
    net.to("cuda")
    print(f"Input tensor's shape:{x.shape}\n")
    print(net)
    y = net(x)
    print(f"Output tensor's shape:{y.shape}")
