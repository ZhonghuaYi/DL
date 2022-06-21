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
    def square_loss(Y_hat, Y):
        m = len(Y)
        return (Y_hat - Y) ** 2. / (2. * m)


class GradDecent:
    @staticmethod
    def square_loss_decent(model, lr, train_X, train_Y, train_Y_hat):
        m = len(train_Y_hat)
        w_grad = np.sum((train_Y_hat - train_Y) * train_X, axis=0) / m
        b_grad = np.sum((train_Y_hat - train_Y), axis=0) / m
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
            GradDecent.square_loss_decent(model, lr, train_X, train_Y, train_Y_hat)
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
            GradDecent.square_loss_decent(model, lr, train_X, train_Y, train_Y_hat)
        return losses


class Model:
    def __init__(self, true_w, true_b):
        self.train_set = DataSet()
        self.valid_set = DataSet()
        self.test_set = DataSet()
        self.w = np.asarray(true_w)
        self.b = np.asarray(true_b)
        self.w_hat = np.random.normal(0, 0.1, self.w.shape)
        self.b_hat = np.random.normal(0, 0.1, self.b.shape)

    def data_generate(self, num):
        """产生数据"""
        X = np.random.normal(2, 1.4, (num, len(self.w)))
        Y = np.matmul(X, self.w) + self.b
        Y += np.random.normal(0, 0.5, Y.shape)
        Y = Y.reshape((-1, 1))
        train = (X[0:int(num * 0.6), ...], Y[0:int(num * 0.6), :])
        valid = (X[int(num * 0.6):int(num * 0.8), ...], Y[int(num * 0.6):int(num * 0.8), :])
        test = (X[int(num * 0.8):num, ...], Y[int(num * 0.8):num, :])
        self.train_set = DataSet(train[0], train[1])
        self.valid_set = DataSet(valid[0], valid[1])
        self.test_set = DataSet(test[0], test[1])

    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        x_parameter, y_parameter = self.train_set.data_scale()
        for data in (self.valid_set, self.test_set):
            data.data_scale(x_parameter, y_parameter)
        w = self.w
        b = self.b
        self.w = x_parameter[1] * w.T / y_parameter[1]
        self.b = (np.sum(x_parameter[0] * w.T) + b - y_parameter[0]) / y_parameter[1]

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

    def predict(self, label="train_set", X=None):
        """根据X预测Y"""
        if X is None:
            data = self.__dict__[label]
            Y_hat = np.matmul(data.X, self.w_hat) + self.b_hat
        else:
            Y_hat = np.matmul(X, self.w_hat) + self.b_hat
        return Y_hat.reshape((-1, 1))

    def train(self, lr, epochs, loss_func=Loss.square_loss, training=Train.batch_train):
        # 最初的参数
        print(f"Initial value:\nw:{self.w} b:{self.b}\n")
        # 开始训练，获得losses
        losses = training(self, loss_func, lr, epochs)
        # 训练集最终loss
        loss = np.sum(loss_func(self.predict("train_set"), self.train_set.Y))
        print(f"End Loss: {loss}")
        # 梯度下降后得到的结果
        print(f"w_hat:{self.w_hat}\nb_hat:{self.b_hat}")
        # 验证集loss
        valid_loss = np.sum(Loss.square_loss(self.predict("valid_set"), self.valid_set.Y))
        print(f"Validation Set Loss: {valid_loss}")
        # 测试集loss
        test_loss = np.sum(Loss.square_loss(self.predict("test_set"), self.test_set.Y))
        print(f"Test Set Loss: {test_loss}")
        # loss曲线
        plt.plot(np.array(list(range(len(losses)))), np.array(losses))
        plt.show()


if __name__ == '__main__':
    linear_model = Model([2.1, 1.4], 0.5)
    linear_model.data_generate(1000)
    linear_model.data_scale()
    linear_model.train(0.1, 1000, training=Train.sgd)
