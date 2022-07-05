# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/4/2022
# @Time  :  10:09 PM

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
import random

from python.template_torch import *


class LinearData(DataSet):
    def __init__(self, sample_num, feature_num, w, b):
        features = torch.normal(2, 1.4, (sample_num, feature_num))
        labels = features @ w + b
        labels += torch.normal(0, 0.2, labels.shape)
        self.features = features
        self.labels = labels

    def data_plot(self, flag="2d"):
        """绘制数据点图，可以是3d"""
        if flag == "2d":
            plt.scatter(self.features[:, 0], self.labels, s=10, c='r')
        elif flag == "3d":
            ax = plt.axes(projection="3d")
            ax.scatter3D(self.features[:, 0], self.features[:, 1], self.labels, s=10, c='r')

    def data_scale(self, x_parameter=None, y_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None and y_parameter is not None:
            x_mean, x_maximum = x_parameter
            y_mean, y_maximum = y_parameter
            self.features = (self.features - x_mean) / x_maximum
            self.labels = (self.labels - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)

        elif x_parameter is None and y_parameter is None:
            x_mean = self.features.mean(dim=1)
            x_maximum = self.features.max(dim=1).values
            self.features = (self.features - x_mean) / x_maximum
            y_mean = self.labels.mean(dim=1)
            y_maximum = self.labels.max(dim=1).values
            self.labels = (self.labels - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)


class LinearNet(nn.Module):
    def __init__(self, feature_num, label_num):
        super().__init__()
        self.fc1 = nn.Linear(feature_num, label_num)

    def forward(self, X):
        Y_hat = self.fc1(X)
        return Y_hat


def batch(X, Y, net, loss, lr, epochs):
    losses = []
    for epoch in range(epochs):
        Y_hat = net(X).reshape(Y.shape)
        l = loss(Y, Y_hat)
        losses.append(l.detach())
        l.backward()
        with torch.no_grad():
            for param in net.parameters():
                param.data.sub_(lr * param.grad)
            net.zero_grad()
    return losses


if __name__ == '__main__':
    w = torch.tensor([1.2, -0.8])
    b = torch.tensor([0.7])

    train_set = LinearData(600, 2, w, b)
    valid_set = LinearData(200, 2, w, b)
    test_set = LinearData(200, 2, w, b)
    train_set.data_plot("3d")
    plt.show()

    lr = 0.02
    epochs = 100
    net = LinearNet(2, 1)
    loss = nn.MSELoss()

    losses = batch(train_set.features, train_set.labels, net, loss, lr, epochs)
    for param in net.parameters():
        print(param)
    plt.plot(np.array(range(len(losses))), np.array(losses))
    plt.show()
