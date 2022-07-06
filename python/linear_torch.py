# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/4/2022
# @Time  :  10:09 PM

import numpy as np
import matplotlib.pyplot as plt

from python.template_torch import *


class LinearData(DataSet):
    def __init__(self, sample_num, feature_num, w, b):
        features = torch.normal(2, 1.4, (sample_num, feature_num))
        labels = features @ w + b
        labels += torch.normal(0, 0.2, labels.shape)
        self.features = features
        self.labels = labels

    def data_plot(self, flag="2d", c='r'):
        """绘制数据点图，可以是3d"""
        if flag == "2d":
            plt.scatter(self.features[:, 0], self.labels, s=10, c=c)
        elif flag == "3d":
            fig = plt.figure("3d data")
            if len(fig.axes) == 0:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.axes[0]
            ax.scatter(self.features[:, 0], self.features[:, 1], self.labels, s=10, c=c)

    def data_scale(self, x_parameter=None, y_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None and y_parameter is not None:
            x_mean, x_maximum = x_parameter
            y_mean, y_maximum = y_parameter
            self.features = (self.features - x_mean) / x_maximum
            self.labels = (self.labels - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)

        elif x_parameter is None and y_parameter is None:
            x_mean = self.features.mean(dim=0)
            x_maximum = self.features.max(dim=0).values
            self.features = (self.features - x_mean) / x_maximum
            y_mean = self.labels.mean(dim=0)
            y_maximum = self.labels.max(dim=0).values
            self.labels = (self.labels - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)


class LinearNet(Net):
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
    train = TrainMethod.mini_batch
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
