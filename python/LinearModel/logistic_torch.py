# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/5/2022
# @Time  :  7:57 PM

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from python.dataset import BinaryClassData
import python.trainmethod as trainmethod


def sigmoid(x):
    if type(x) == torch.Tensor:
        return 1 / (1 + torch.exp(-x))
    elif type(x) == np.ndarray:
        return 1 / (1 + np.exp(-x))


class CrossEntropyLoss:
    @staticmethod
    def loss(Y_hat, Y):
        m = Y.shape[0]
        return - torch.sum((Y * torch.log(Y_hat) + (1 - Y) * torch.log(1 - Y_hat)) / m)

    @staticmethod
    def regularized_loss(Y_hat, Y, w_hat, b_hat, lamda):
        m = Y.shape[0]
        regular_item = lamda / (2 * m) * (torch.sum(w_hat ** 2, dim=1) +
                                          torch.sum(b_hat ** 2, dim=1))
        return CrossEntropyLoss.loss(Y_hat, Y) + regular_item


class LogisticNet(nn.Module):
    def __init__(self, feature_num, label_num):
        super().__init__()
        self.fc1 = nn.Linear(feature_num, label_num)

    def forward(self, X):
        Y_hat = sigmoid(self.fc1(X))
        return Y_hat


if __name__ == '__main__':
    train_set = BinaryClassData(300, 300)
    test_set = BinaryClassData(200, 200)

    lr = 0.1
    epochs = 10
    batch_size = 50
    net = LogisticNet(2, 1)
    loss = CrossEntropyLoss.loss
    train = trainmethod.mini_batch
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)

    losses = train(train_set, net, loss, trainer, epochs, batch_size)
    plt.plot(np.array(range(len(losses))), np.array(losses))
    plt.show()

    with torch.no_grad():
        # 画出数据
        train_set.data_plot("3d")
        test_set.data_plot("3d", c='y')
        # 画出用于预测的直线（平面）
        x = np.linspace(-3, 3, 100)
        y = x
        x, y = np.meshgrid(x, y)
        w_hat, b_hat = list(net.fc1.parameters())
        print(f"w_hat: {w_hat.numpy()}")
        print(f"b_hat: {b_hat.numpy()}")
        z = w_hat[0, 0] * x + w_hat[0, 1] * y + b_hat
        fig = plt.figure("3d data")
        ax = fig.axes[0]
        # ax.plot_surface(x, y, z, color=(0, 1, 0, 0.3))
        # 画出直线（平面）经过sigmoid拟合后预测的概率
        ax.plot_surface(x, y, sigmoid(z), color=(0, 0, 1, 0.3))
        ax.set_zlim(-0.5, 1.5)
        plt.show()
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
