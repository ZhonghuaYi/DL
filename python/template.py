# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  2022-06-28
# @Time  :  2:56 PM

import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import abc


class DataSet(metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def data_scale(self):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        pass


class Loss(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def loss(Y_hat, Y):
        pass


class GradDecent(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def decent(model, lr, train_X, train_Y, train_Y_hat):
        pass


class Train:
    @staticmethod
    def batch_train(model, loss_func, decent, lr, epochs):
        losses = []
        for epoch in range(epochs):
            train_X = model.train_set.X
            train_Y = model.train_set.Y
            train_Y_hat = model.predict(train_X)
            loss = loss_func(train_Y_hat, model.train_set.Y)
            losses.append(loss)
            # print(f"loss before epoch{epoch}: {loss}")
            decent(model, lr, train_X, train_Y, train_Y_hat)
        return losses

    @staticmethod
    def sgd(model, loss_func, decent, lr):
        losses = []
        indies = list(range(len(model.train_set.Y)))
        random.shuffle(indies)
        for i in indies:
            train_X = model.train_set.X[i, ...]
            train_Y = model.train_set.Y[i]
            train_Y_hat = model.predict(train_X)
            loss = loss_func(train_Y_hat, model.train_set.Y[i])
            if i % 10 == 0:
                losses.append(loss)
                # print(f"loss before epoch{epoch}: {loss}")
            decent(model, lr, train_X, train_Y, train_Y_hat)
        return losses


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        self.train_set = DataSet()
        self.valid_set = DataSet()
        self.test_set = DataSet()

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def data_generate(self, *args):
        """产生数据"""
        pass

    @abc.abstractmethod
    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        pass

    def data_plot(self, flag="2d"):
        """绘制训练集、验证集、测试集在一个图中"""
        if flag == "2d":
            plt.scatter(self.train_set.X[:, 0], self.train_set.Y, s=10, c='r')
            plt.scatter(self.valid_set.X[:, 0], self.valid_set.Y, s=10, c='b')
            plt.scatter(self.test_set.X[:, 0], self.test_set.Y, s=10, c='y')
        elif flag == "3d":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(self.train_set.X[:, 0],
                         self.train_set.X[:, 1],
                         self.train_set.Y,
                         s=10,
                         c='r')
            ax.scatter(self.valid_set.X[:, 0],
                         self.valid_set.X[:, 1],
                         self.valid_set.Y,
                         s=10,
                         c='b')
            ax.scatter(self.test_set.X[:, 0],
                         self.test_set.X[:, 1],
                         self.test_set.Y,
                         s=10,
                         c='y')
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Label")

    @abc.abstractmethod
    def predict(self, X):
        """根据X预测Y"""
        pass

    def train(self, lr, epochs, loss_func, decent, training):
        # 最初的参数
        print(f"Initial value:\n{self}\n")
        # 开始训练，获得losses
        losses = training(self, loss_func, decent, lr, epochs)
        # 训练集最终loss
        loss = loss_func(self.predict(self.train_set.X), self.train_set.Y)
        print(f"End Loss: {loss}")
        # 梯度下降后得到的结果
        print(f"Result:\n{self}\n")
        # 验证集loss
        valid_loss = loss_func(self.predict(self.valid_set.X), self.valid_set.Y)
        print(f"Validation Set Loss: {valid_loss}")
        # 测试集loss
        test_loss = loss_func(self.predict(self.test_set.X), self.test_set.Y)
        print(f"Test Set Loss: {test_loss}")
        # loss曲线
        plt.plot(np.array(list(range(len(losses)))), np.array(losses))
        plt.title("Loss-Epoch relation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
