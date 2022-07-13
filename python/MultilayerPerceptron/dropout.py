# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/12/2022
# @Time  :  9:30 PM

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from python.dataset import MnistData, FashionMnistData
import python.trainmethod as trainmethod


def dropout(X, p):
    assert 0<=p<=1
    if p == 0:
        return X
    elif p == 1:
        return torch.zeros(X.shape)
    else:
        mask = (torch.rand(X.shape) > p).float()
        X = X * mask / (1-p)
    return X


class DropoutNet(nn.Module):
    def __init__(self, feature_num, label_num):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(feature_num, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, label_num)

    def forward(self, X):
        Y_hat = self.fc1(X)
        Y_hat = self.relu(Y_hat)
        # 在隐藏层后加入dropout层
        Y_hat = dropout(Y_hat, 0.5)
        Y_hat = self.fc2(Y_hat)
        return Y_hat


if __name__ == '__main__':
    train_set = FashionMnistData("train")
    test_set = FashionMnistData("test")
    x_mean, x_maximum = train_set.data_scale()
    test_set.data_scale((x_mean, x_maximum))

    net = DropoutNet(784, 10)
    lr = 0.1
    epochs = 10
    batch_size = 50
    loss = nn.CrossEntropyLoss()
    train = trainmethod.mini_batch
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)

    losses = train(train_set, net, loss, trainer, epochs, batch_size)
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
