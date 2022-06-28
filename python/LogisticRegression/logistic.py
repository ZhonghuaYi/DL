# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-06-26

import numpy as np
import random
from matplotlib import pyplot as plt

from python.template import *


def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticData(DataSet):
    """逻辑回归的数据"""

    def data_scale(self, x_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None:
            x_mean, x_maximum = x_parameter
            self.X = (self.X - x_mean) / x_maximum
            return x_mean, x_maximum

        elif x_parameter is None:
            x_mean = np.mean(self.X, axis=0)
            x_maximum = np.max(self.X, axis=0)
            self.X = (self.X - x_mean) / x_maximum
            return x_mean, x_maximum


class CrossEntropyLoss(Loss):
    @staticmethod
    def loss(Y_hat, Y):
        m = len(Y)
        return - np.sum((Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat)) / m)


class CrossEntropyLossDecent:
    @staticmethod
    def decent(model, lr, X, Y, Y_hat):
        m = len(Y_hat)
        w_grad = - np.sum(X*(Y*(1-Y_hat)-Y_hat*(1-Y)), axis=0) / m
        b_grad = - np.sum(Y*(1-Y_hat)-Y_hat*(1-Y), axis=0) / m
        model.w_hat = model.w_hat - lr * w_grad
        model.b_hat = model.b_hat - lr * b_grad


class LogisticTrain(Train):
    pass


class LogisticModel(Model):
    def __init__(self):
        self.train_set = LogisticData()
        self.valid_set = LogisticData()
        self.test_set = LogisticData()
        self.w_hat = np.random.normal(0, 0.1, 1)
        self.b_hat = np.random.normal(0, 0.1, 1)

    def __str__(self):
        return f"w_hat: {self.w_hat} b_hat: {self.b_hat}"

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
        self.train_set = LogisticData(train[0], train[1])
        self.valid_set = LogisticData(valid[0], valid[1])
        self.test_set = LogisticData(test[0], test[1])

    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        x_parameter= self.train_set.data_scale()
        for data in (self.valid_set, self.test_set):
            data.data_scale(x_parameter)

    def data_plot(self, flag="2d"):
        """绘制训练集、验证集、测试集在一个图中"""
        super(LogisticModel, self).data_plot(flag)

    def predict(self, X):
        """根据X预测Y"""
        Y_hat = sigmoid(np.matmul(X, self.w_hat) + self.b_hat)
        return Y_hat.reshape((-1, 1))


if __name__ == '__main__':
    logistic = LogisticModel()
    logistic.data_generate(500, 500)
    logistic.train(1, 100, CrossEntropyLoss.loss, CrossEntropyLossDecent.decent, LogisticTrain.batch_train)
    logistic.data_plot()
    x = np.linspace(-3, 3, 100)
    y = logistic.w_hat * x + logistic.b_hat
    plt.plot(x, y, c='y')
    plt.plot(x, sigmoid(y), c='r')
    plt.show()
