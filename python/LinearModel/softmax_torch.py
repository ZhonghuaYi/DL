# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/6/2022
# @Time  :  11:00 PM

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from python.dataset import MnistData, one_hot
import python.trainmethod as trainmethod


def softmax(x):
    if type(x) == torch.Tensor:
        x_exp = torch.exp(x)
        row_sum = 0
        if x.ndim == 1:
            row_sum = x_exp.sum()
        elif x.ndim == 2:
            row_sum = x_exp.sum(dim=1).reshape(-1, 1)
        return x_exp / row_sum
    elif type(x) == np.ndarray:
        x_exp = np.exp(x)
        row_sum = 0
        if x.ndim == 1:
            row_sum = np.sum(x_exp)
        if x.ndim == 2:
            row_sum = np.sum(x_exp, axis=1).reshape(-1, 1)
        return x_exp / row_sum


class SoftmaxNet(nn.Module):
    def __init__(self, feature_num, label_num):
        super().__init__()
        self.fc1 = nn.Linear(feature_num, label_num)

    def forward(self, X):
        Y_hat = self.fc1(X)
        return Y_hat


if __name__ == '__main__':
    train_set = MnistData("train")
    test_set = MnistData("test")
    x_mean, x_maximum = train_set.data_scale()
    test_set.data_scale((x_mean, x_maximum))

    lr = 0.1
    epochs = 10
    batch_size = 50
    net = SoftmaxNet(784, 10)
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
