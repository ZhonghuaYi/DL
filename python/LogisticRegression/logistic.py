# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-06-26

import numpy as np
import random
from matplotlib import pyplot as plt

from python.template import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        return - np.sum((Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / m)

    @staticmethod
    def regularized_loss(model, Y_hat, Y, lamda):
        m = len(Y)
        regular_item = lamda / (2 * m) * (np.sum(model.w_hat ** 2, axis=0) +
                                          np.sum(model.b_hat ** 2, axis=0))
        return CrossEntropyLoss.loss(Y_hat, Y) + regular_item


class CrossEntropyLossDecent(GradDecent):
    @staticmethod
    def decent(model, lr, X, Y, Y_hat):
        m = len(Y_hat)
        w_grad = - np.sum(X * (Y * (1 - Y_hat) - Y_hat * (1 - Y)), axis=0) / m
        b_grad = - np.sum(Y * (1 - Y_hat) - Y_hat * (1 - Y), axis=0) / m
        model.w_hat = model.w_hat - lr * w_grad
        model.b_hat = model.b_hat - lr * b_grad

    @staticmethod
    def regular_decent(model, lr, lamda, train_X, train_Y, train_Y_hat):
        m = len(train_Y_hat)
        w_grad = np.sum((train_Y_hat - train_Y) * train_X, axis=0) / m
        b_grad = np.sum((train_Y_hat - train_Y), axis=0) / m
        model.w_hat = model.w_hat - lr * (w_grad + lamda / m * w_grad)
        model.b_hat = model.b_hat - lr * (b_grad + lamda / m * b_grad)


class LogisticTrain(Train):
    pass


class LogisticModel(Model):
    def __init__(self):
        self.train_set = LogisticData()
        self.valid_set = LogisticData()
        self.test_set = LogisticData()
        self.w_hat = np.random.normal(0, 0.1, 2)
        self.b_hat = np.random.normal(0, 0.1, 1)

    def __str__(self):
        return f"w_hat: {self.w_hat} b_hat: {self.b_hat}"

    def data_generate(self, num1, num2):
        num = num1 + num2
        X11 = np.random.normal(-2, 1, (num1, 1))
        X12 = np.random.normal(2, 1, (num1, 1))
        Y1 = np.zeros((num1, 1))
        X21 = np.random.normal(2, 1, (num2, 1))
        X22 = np.random.normal(-2, 1, (num2, 1))
        Y2 = np.ones((num2, 1))
        Z = np.hstack((np.vstack((X11, X21)), np.vstack((X12, X22)), np.vstack((Y1, Y2))))
        np.random.shuffle(Z)
        X = Z[:, 0:-1]
        Y = Z[:, -1].reshape(-1, 1)
        train = (X[0:int(num * 0.6), ...], Y[0:int(num * 0.6), :])
        valid = (X[int(num * 0.6):int(num * 0.8), ...], Y[int(num * 0.6):int(num * 0.8), :])
        test = (X[int(num * 0.8):num, ...], Y[int(num * 0.8):num, :])
        self.train_set = LogisticData(train[0], train[1])
        self.valid_set = LogisticData(valid[0], valid[1])
        self.test_set = LogisticData(test[0], test[1])

    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        x_parameter = self.train_set.data_scale()
        for data in (self.valid_set, self.test_set):
            data.data_scale(x_parameter)

    def data_plot(self, flag="2d"):
        """绘制训练集、验证集、测试集在一个图中"""
        super(LogisticModel, self).data_plot(flag)

    def predict(self, X):
        """根据X预测Y"""
        # print(X.shape, self.w_hat.shape)
        Y_hat = sigmoid(np.matmul(X, self.w_hat) + self.b_hat)
        return Y_hat.reshape((-1, 1))

    def accuracy(self, label="train_set"):
        """计算正确率"""
        data = self.__dict__[label]
        result = (self.predict(data.X) > 0.5).astype(data.Y.dtype)
        accuracy_rate = np.sum((result == data.Y).astype(np.uint8)) / len(data.Y)
        print(f"Accuracy of {label}: {accuracy_rate}")

    def precision(self, label="train_set"):
        """计算查全率"""
        data = self.__dict__[label]
        result = (self.predict(data.X) > 0.5).astype(data.Y.dtype)
        result_1 = result * data.Y
        precision_rate = np.sum(result_1) / np.sum(data.Y)
        print(f"Precision of {label}: {precision_rate}")

    def recall(self, label="train_set"):
        """计算查全率"""
        data = self.__dict__[label]
        result = (self.predict(data.X) > 0.5).astype(data.Y.dtype)
        result_1 = result * data.Y
        recall_rate = np.sum(result_1) / np.sum(result)
        print(f"Precision of {label}: {recall_rate}")


if __name__ == '__main__':
    # 设置训练的参数
    train_dict = {
        "lr": 0.1,
        "epochs": 1000,
        "loss_func": CrossEntropyLoss.loss,
        "decent": CrossEntropyLossDecent.decent,
        "regularized_loss_func": CrossEntropyLoss.regularized_loss,
        "regular_decent": CrossEntropyLossDecent.regular_decent,
        "lamda": 2,
        "training": LogisticTrain.batch_train,
    }
    train_parameter = TrainParameter()
    train_parameter.set(**train_dict)

    logistic = LogisticModel()  # 生成模型实例
    logistic.data_generate(500, 500)  # 生成模型的数据

    logistic.train(train_parameter)  # 训练参数
    logistic.valid(train_parameter.loss_func)  # 计算验证集的loss
    logistic.test(train_parameter.loss_func)  # 计算测试集的loss

    # 计算训练集、验证集和测试集的正确率、查准率和查全率
    for data_set in ("train_set", "valid_set", "test_set"):
        print(f"{data_set}:")
        logistic.accuracy(data_set)
        logistic.precision(data_set)
        logistic.recall(data_set)

    # 画出数据
    logistic.data_plot("3d")
    # 画出用于预测的直线（平面）
    x = np.linspace(-3, 3, 100)
    y = x
    x, y = np.meshgrid(x, y)
    z = logistic.w_hat[0] * x + logistic.w_hat[1] * y + logistic.b_hat
    fig = plt.figure("3d data")
    ax = fig.axes[0]
    ax.plot_surface(x, y, z, color=(0, 1, 0, 0.3))
    # 画出直线（平面）经过sigmoid拟合后预测的概率
    ax.plot_surface(x, y, sigmoid(z), color=(0, 0, 1, 0.3))
    # 显示数据图像
    plt.show()
