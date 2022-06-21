# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-06-21

import numpy as np
import random
from matplotlib import pyplot as plt


class DataSet:
    """线性回归的数据"""
    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y

    def data_iter(self, batch_size):
        """产生数据子集"""
        nums = len(self.Y)
        index = list(range(nums))
        random.shuffle(index)
        for i in range(0, nums, batch_size):
            ind = index[i:min(i+batch_size, nums)]
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


class Model:
    def __init__(self, true_w, true_b):
        self.train = DataSet()
        self.valid = DataSet()
        self.test = DataSet()
        self.w = np.array(true_w)
        self.b = np.array(true_b)
        self.w_hat = np.random.normal(0, 0.1, self.w.shape)
        self.b_hat = np.random.normal(0, 0.1, self.b.shape)

    def data_generate(self, num):
        """产生数据"""
        X = np.random.normal(0, 1.4, (num, len(self.w)))
        Y = np.matmul(X, self.w) + self.b
        Y += np.random.normal(0, 0.5, Y.shape)
        Y = Y.reshape((-1, 1))
        train = (X[0:int(num*0.6), ...], Y[0:int(num*0.6), :])
        valid = (X[int(num*0.6):int(num*0.8), ...], Y[int(num*0.6):int(num*0.8), :])
        test = (X[int(num*0.8):num, ...], Y[int(num*0.8):num, :])
        self.train = DataSet(train[0], train[1])
        self.valid = DataSet(valid[0], valid[1])
        self.test = DataSet(test[0], test[1])

    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        x_parameter, y_parameter = self.train.data_scale()
        for data in (self.valid, self.test):
            data.data_scale(x_parameter, y_parameter)

    def data_plot(self, flag="2d"):
        """绘制训练集、验证集、测试集在一个图中"""
        if flag == "2d":
            plt.scatter(self.train.X[:, 0], self.train.Y, s=10, c='r')
            plt.scatter(self.valid.X[:, 0], self.valid.Y, s=10, c='b')
            plt.scatter(self.test.X[:, 0], self.test.Y, s=10, c='y')
        elif flag == "3d":
            ax = plt.axes(projection="3d")
            ax.scatter3D(self.train.X[:, 0], self.train.X[:, 1], self.train.Y, s=10, c='r')
            ax.scatter3D(self.valid.X[:, 0], self.valid.X[:, 1], self.valid.Y, s=10, c='b')
            ax.scatter3D(self.test.X[:, 0], self.test.X[:, 1], self.test.Y, s=10, c='y')

    def predict(self, label="train"):
        """根据X预测Y"""
        data = self.__dict__[label]
        Y_hat = np.matmul(data.X, self.w_hat) + self.b_hat
        return Y_hat


class Train:
    def __init__(self, model, lr):
        pass


if __name__ == '__main__':
    model = Model([2.1, 1.3], 0.5)
    model.data_generate(1000)
    model.data_scale()
    model.data_plot("3d")
    plt.show()
