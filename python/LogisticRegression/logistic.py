# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-06-26

import numpy as np
import random
from matplotlib import pyplot as plt


class DataSet:
    """逻辑回归的数据"""

    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y

    def data_iter(self, batch_size):
        """产生数据子集"""
        nums = len(self.Y)
        index = list(range(nums))
        random.shuffle(index)
        for i in range(0, nums, batch_size):
            ind = index[i:min(i + batch_size, nums)]
            yield self.X[ind], self.Y[ind]

    def data_plot(self, flag="2d"):
        """绘制数据点图，可以是3d"""
        if flag == "2d":
            plt.scatter(self.X[:, 0], self.Y, s=10, c='r')
        elif flag == "3d":
            ax = plt.axes(projection="3d")
            ax.scatter3D(self.X[:, 0], self.X[:, 1], self.Y, s=10, c='r')

    def data_scale(self, x_parameter=None, y_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None and y_parameter is not None:
            x_mean, x_maximum = x_parameter
            y_mean, y_maximum = y_parameter
            self.X = (self.X - x_mean) / x_maximum
            self.Y = (self.Y - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)

        elif x_parameter is None and y_parameter is None:
            x_mean = np.mean(self.X, axis=0)
            x_maximum = np.max(self.X, axis=0)
            self.X = (self.X - x_mean) / x_maximum
            y_mean = np.mean(self.Y, axis=0)
            y_maximum = np.max(self.Y, axis=0)
            self.Y = (self.Y - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)


class Loss:
    @staticmethod
    def cross_entropy_loss(Y_hat, Y):
        m = len(Y)
        return - (Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat)) / m


class GradDecent:
    @staticmethod
    def cross_entropy_loss_decent(model, lr, train_X, train_Y, train_Y_hat):
        m = len(train_Y_hat)
        pass


class Train:
    pass


class Model:
    def __init__(self):
        pass

    def data_generate(self, num1, num2):
        pass


if __name__ == '__main__':
    pass
