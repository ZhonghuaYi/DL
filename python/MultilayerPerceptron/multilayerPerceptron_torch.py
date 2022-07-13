# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/8/2022
# @Time  :  1:39 PM

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from python.dataset import MnistData, FashionMnistData
import python.trainmethod as trainmethod


class PerceptronNet(nn.Module):
    def __init__(self, feature_num, label_num):
        super(PerceptronNet, self).__init__()
        # 但损失函数使用了CrossEntropyLoss时，训练时网络的最后不需要执行softmax，但是预测时需要
        self.fc = nn.Sequential(nn.Linear(feature_num, 256),
                                nn.ReLU(),
                                nn.Linear(256, label_num))

    def forward(self, X):
        Y_hat = self.fc(X)
        return Y_hat


if __name__ == '__main__':
    train_set = FashionMnistData("train")
    test_set = FashionMnistData("test")
    x_mean, x_maximum = train_set.data_scale()
    test_set.data_scale((x_mean, x_maximum))

    net = PerceptronNet(784, 10)
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
