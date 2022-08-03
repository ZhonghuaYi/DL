# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/13/2022
# @Time  :  7:14 PM

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


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(),
                                  nn.AvgPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(6, 16, 5), nn.Sigmoid(),
                                  nn.AvgPool2d(kernel_size=2, stride=2))
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(16*5*5, 120), nn.Sigmoid(),
                                    nn.Linear(120, 84), nn.Sigmoid(),
                                    nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear(x)
        return x


def init_params(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


if __name__ == '__main__':
    train_set = FashionMnistData("train")
    test_set = FashionMnistData("test")
    x_mean, x_maximum = train_set.data_scale()
    test_set.data_scale((x_mean, x_maximum))
    train_set = reshape(train_set)
    test_set = reshape(test_set)

    net = LeNet()
    # net.apply(init_params)
    lr = 0.001
    epochs = 5
    batch_size = 50
    device = "cuda"
    device_num = 0
    loss = nn.CrossEntropyLoss()
    train = trainmethod.mini_batch
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)

    losses = train(train_set, net, loss, trainer, epochs, batch_size, device=device, device_num=device_num)
    # print(losses)
    plt.plot(np.array(range(len(losses))), np.array(losses))
    plt.show()

    with torch.no_grad():
        # 训练集的正确率、查准率和查全率
        print("\nTrain set:")
        train_set.accuracy(net)
        train_set.precision(net)
        train_set.recall(net)
        # 测试集的正确率、查准率和查全率
        print("\nTest set:")
        test_set.accuracy(net)
        test_set.precision(net)
        test_set.recall(net)
