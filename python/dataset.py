# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/12/2022
# @Time  :  10:33 PM

import abc
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def one_hot(Y):
    if type(Y) == torch.Tensor:
        y = Y
        m = y.shape[0]
        new_y = torch.zeros((m, 10), dtype=torch.int32, device=y.device)
        for i in range(m):
            new_y[i, int(y[i])] = 1
        return new_y
    elif type(Y) == np.ndarray:
        y = Y
        m = y.shape[0]
        new_y = np.zeros((m, 10), dtype=np.int32)
        for i in range(m):
            new_y[i, int(y[i])] = 1
        return new_y


def softmax(x):
    if type(x) == torch.Tensor:
        x_exp = torch.exp(x)
        row_sum = 0
        if x.ndim == 1:
            row_sum = x_exp.sum()
        elif x.ndim == 2:
            row_sum = x_exp.sum(dim=1).reshape(-1, 1)
        return x_exp / row_sum
    elif type(x) == np.ndarray:
        x_exp = np.exp(x)
        row_sum = 0
        if x.ndim == 1:
            row_sum = np.sum(x_exp)
        if x.ndim == 2:
            row_sum = np.sum(x_exp, axis=1).reshape(-1, 1)
        return x_exp / row_sum


class DataSet(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        self.features = None
        self.labels = None

    def __getitem__(self, item):
        return self.features[item, ...], self.labels[item, ...]

    def __len__(self):
        return self.labels.shape[0]

    def data_iter(self, batch_size):
        nums = self.labels.shape[0]
        index = list(range(nums))
        random.shuffle(index)
        for i in range(0, nums, batch_size):
            ind = index[i:min(i + batch_size, nums)]
            yield self.features[ind], self.labels[ind]

    @abc.abstractmethod
    def data_plot(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def data_scale(self, *args, **kwargs):
        pass


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


class BinaryClassData(DataSet):
    def __init__(self, num1, num2):
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
        self.features = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(Y).float()

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

    def data_scale(self, x_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None:
            x_mean, x_maximum = x_parameter
            self.features = (self.features - x_mean) / x_maximum
            return x_mean, x_maximum

        elif x_parameter is None:
            x_mean = self.features.mean(dim=0)
            x_maximum = self.features.max(dim=0).values
            self.features = (self.features - x_mean) / x_maximum
            return x_mean, x_maximum

    def accuracy(self, net):
        """计算正确率"""
        with torch.no_grad():
            result = (net(self.features) > 0.5).clone().detach().reshape(self.labels.shape).int()
            accuracy_rate = (result == self.labels).sum() / self.labels.shape[0]
            print(f"Accuracy: {accuracy_rate}")

    def precision(self, net):
        """计算查准率"""
        with torch.no_grad():
            result = (net(self.features) > 0.5).clone().detach().reshape(self.labels.shape).int()
            result_1 = result * self.labels
            precision_rate = result_1.sum() / self.labels.sum()
            print(f"Precision: {precision_rate}")

    def recall(self, net):
        """计算查全率"""
        with torch.no_grad():
            result = (net(self.features) > 0.5).clone().detach().reshape(self.labels.shape).int()
            result_1 = result * self.labels
            recall_rate = result_1.sum() / result.sum()
            print(f"Recall: {recall_rate}")


class MnistData(DataSet):
    def __init__(self, label="train"):
        from python.data.mnist import load_mnist
        train_images, train_labels, test_images, test_labels = load_mnist("../data/")
        if label == "train":
            self.features = torch.from_numpy(train_images[:60000, ...]).float()
            self.labels = torch.from_numpy(one_hot(train_labels[:60000, ...])).float()
        elif label == "test":
            self.features = torch.from_numpy(test_images[:10000, ...]).float()
            self.labels = torch.from_numpy(one_hot(test_labels[:10000, ...])).float()

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

    def data_scale(self, x_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None:
            mask = self.features == 0
            x_mean, x_maximum = x_parameter
            self.features = (self.features - x_mean) / x_maximum
            self.features[mask] = 0
            return x_mean, x_maximum

        elif x_parameter is None:
            mask = self.features == 0
            x_mean = self.features.mean(dim=0)
            x_maximum = self.features.max(dim=0).values
            self.features = (self.features - x_mean) / x_maximum
            self.features[mask] = 0
            return x_mean, x_maximum

    def accuracy(self, net):
        """计算正确率"""
        with torch.no_grad():
            result = one_hot(softmax(net(self.features)).argmax(dim=1))
            accuracy_rate = (result * self.labels).sum() / self.labels.shape[0]
            print(f"Accuracy: {accuracy_rate}")

    def precision(self, net):
        """计算查准率"""
        with torch.no_grad():
            result = one_hot(softmax(net(self.features)).argmax(dim=1))
            result_1 = result * self.labels
            precision_rate = result_1.sum(dim=0) / self.labels.sum(dim=0)
            print(f"Precision: {precision_rate}")

    def recall(self, net):
        """计算查全率"""
        with torch.no_grad():
            result = one_hot(softmax(net(self.features)).argmax(dim=1))
            result_1 = result * self.labels
            recall_rate = result_1.sum(dim=0) / result.sum(dim=0)
            print(f"Recall: {recall_rate}")


class FashionMnistData(DataSet):
    def __init__(self, label="train"):
        from python.data.fashion_mnist import load_fashion_mnist
        train_images, train_labels, test_images, test_labels = load_fashion_mnist("../data/")
        if label == "train":
            self.features = torch.from_numpy(train_images[:60000, ...]).float()
            self.labels = torch.from_numpy(one_hot(train_labels[:60000, ...])).float()
        elif label == "test":
            self.features = torch.from_numpy(test_images[:10000, ...]).float()
            self.labels = torch.from_numpy(one_hot(test_labels[:10000, ...])).float()

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

    def data_scale(self, x_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None:
            mask = self.features == 0
            x_mean, x_maximum = x_parameter
            self.features = (self.features - x_mean) / x_maximum
            self.features[mask] = 0
            return x_mean, x_maximum

        elif x_parameter is None:
            mask = self.features == 0
            x_mean = self.features.mean(dim=0)
            x_maximum = self.features.max(dim=0).values
            self.features = (self.features - x_mean) / x_maximum
            self.features[mask] = 0
            return x_mean, x_maximum

    def accuracy(self, net):
        """计算正确率"""
        with torch.no_grad():
            result = one_hot(softmax(net(self.features)).argmax(dim=1))
            accuracy_rate = (result * self.labels).sum() / self.labels.shape[0]
            print(f"Accuracy: {accuracy_rate}")

    def precision(self, net):
        """计算查准率"""
        with torch.no_grad():
            result = one_hot(softmax(net(self.features)).argmax(dim=1))
            result_1 = result * self.labels
            precision_rate = result_1.sum(dim=0) / self.labels.sum(dim=0)
            print(f"Precision: {precision_rate}")

    def recall(self, net):
        """计算查全率"""
        with torch.no_grad():
            result = one_hot(softmax(net(self.features)).argmax(dim=1))
            result_1 = result * self.labels
            recall_rate = result_1.sum(dim=0) / result.sum(dim=0)
            print(f"Recall: {recall_rate}")
