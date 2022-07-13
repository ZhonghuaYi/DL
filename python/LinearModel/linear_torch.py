# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/4/2022
# @Time  :  10:09 PM

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from python.dataset import LinearData
import python.trainmethod as trainmethod


class LinearNet(nn.Module):
    def __init__(self, feature_num, label_num):
        super().__init__()
        self.fc1 = nn.Linear(feature_num, label_num)

    def forward(self, X):
        Y_hat = self.fc1(X)
        return Y_hat


if __name__ == '__main__':
    w = torch.tensor([1.2, -0.8])
    b = torch.tensor([0.7])

    train_set = LinearData(600, 2, w, b)
    valid_set = LinearData(200, 2, w, b)
    test_set = LinearData(200, 2, w, b)

    lr = 0.01
    epochs = 10
    batch_size = 50
    net = LinearNet(2, 1)
    loss = nn.MSELoss()
    train = trainmethod.mini_batch
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)

    losses = train(train_set, net, loss, trainer, epochs, batch_size)
    with torch.no_grad():
        # 训练集loss
        print("Loss of train set: ",
              loss(net(train_set.features).reshape(train_set.labels.shape),
                   train_set.labels))
        # 验证集loss
        print("Loss of valid set: ",
              loss(net(valid_set.features).reshape(valid_set.labels.shape),
                   valid_set.labels))
        # 测试集loss
        print("Loss of valid set: ",
              loss(net(test_set.features).reshape(test_set.labels.shape),
                   test_set.labels))
        print(list(net.named_parameters()))
        # 绘制数据的拟合图
        train_set.data_plot("3d")
        valid_set.data_plot("3d", c='b')
        test_set.data_plot("3d", c='y')
        x = np.linspace(-2, 6, 100)
        y = x
        x, y = np.meshgrid(x, y)
        z = net.fc1.weight[0, 0] * x + net.fc1.weight[0, 1] * y + net.fc1.bias[0]
        fig = plt.figure("3d data")
        ax = fig.axes[0]
        ax.plot_surface(x, y, z, color=(0, 1, 0, 0.3))
        plt.show()
        # 绘制loss变化曲线
        plt.plot(np.array(range(len(losses))), np.array(losses))
        plt.show()
