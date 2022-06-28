# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-06-26

import numpy as np
import random
from matplotlib import pyplot as plt

from ..LinearRegression.linear import *


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
    def cross_entropy_loss_decent(model, lr, X, Y, Y_hat):
        m = len(Y_hat)
        w_grad = - np.sum(X*(Y*(1-Y_hat)-Y_hat*(1-Y)), axis=0) / m
        b_grad = - np.sum(Y*(1-Y_hat)-Y_hat*(1-Y), axis=0) / m
        model.w_hat = model.w_hat - lr * w_grad
        model.b_hat = model.b_hat - lr * b_grad


class Train:
    @staticmethod
    def batch_train(model, loss_func, lr, epochs):
        losses = []
        for epoch in range(epochs):
            train_X = model.train_set.X
            train_Y = model.train_set.Y
            train_Y_hat = model.predict("train_set")
            loss = np.sum(loss_func(train_Y_hat, model.train_set.Y))
            losses.append(loss)
            # print(f"loss before epoch{epoch}: {loss}")
            GradDecent.cross_entropy_loss_decent(model, lr, train_X, train_Y, train_Y_hat)
        return losses

    @staticmethod
    def sgd(model, loss_func, lr, epochs):
        losses = []
        indies = list(range(len(model.train_set.Y)))
        random.shuffle(indies)
        for i in indies:
            train_X = model.train_set.X[i, ...]
            train_Y = model.train_set.Y[i]
            train_Y_hat = model.predict(X=train_X)
            loss = np.sum(loss_func(train_Y_hat, model.train_set.Y[i]))
            if i % 10 == 0:
                losses.append(loss)
                # print(f"loss before epoch{epoch}: {loss}")
            GradDecent.cross_entropy_loss_decent(model, lr, train_X, train_Y, train_Y_hat)
        return losses


class Model:
    def __init__(self):
        self.train_set = DataSet()
        self.valid_set = DataSet()
        self.test_set = DataSet()
        self.w_hat = np.random.normal(0, 0.1, (1, 1))
        self.b_hat = np.random.normal(0, 0.1, (1, 1))

    def data_generate(self, num1, num2):
        num = num1 + num2
        X1 = np.random.normal(-2, 0.7, (num1, 1))
        Y1 = np.zeros((num1, 1))
        X2 = np.random.normal(2, 0.7, (num2, 1))
        Y2 = np.ones((num2, 1))
        Z = np.hstack((np.vstack((X1, X2)), np.vstack((Y1, Y2))))
        np.random.shuffle(Z)
        X = Z[:, 0:-1]
        Y = Z[:, -1].reshape(-1, 1)
        print(X.shape, Y.shape)
        train = (X[0:int(num * 0.6), ...], Y[0:int(num * 0.6), :])
        valid = (X[int(num * 0.6):int(num * 0.8), ...], Y[int(num * 0.6):int(num * 0.8), :])
        test = (X[int(num * 0.8):num, ...], Y[int(num * 0.8):num, :])
        self.train_set = DataSet(train[0], train[1])
        self.valid_set = DataSet(valid[0], valid[1])
        self.test_set = DataSet(test[0], test[1])

    def data_plot(self, flag="2d"):
        """绘制训练集、验证集、测试集在一个图中"""
        if flag == "2d":
            plt.scatter(self.train_set.X[:, 0], self.train_set.Y, s=10, c='r')
            plt.scatter(self.valid_set.X[:, 0], self.valid_set.Y, s=10, c='b')
            plt.scatter(self.test_set.X[:, 0], self.test_set.Y, s=10, c='y')
        elif flag == "3d":
            ax = plt.axes(projection="3d")
            ax.scatter3D(self.train_set.X[:, 0],
                         self.train_set.X[:, 1],
                         self.train_set.Y,
                         s=10,
                         c='r')
            ax.scatter3D(self.valid_set.X[:, 0],
                         self.valid_set.X[:, 1],
                         self.valid_set.Y,
                         s=10,
                         c='b')
            ax.scatter3D(self.test_set.X[:, 0],
                         self.test_set.X[:, 1],
                         self.test_set.Y,
                         s=10,
                         c='y')


if __name__ == '__main__':
    logistic = Model()
    logistic.data_generate(500, 500)
    logistic.data_plot()
    plt.show()
