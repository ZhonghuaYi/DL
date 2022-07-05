# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/4/2022
# @Time  :  10:15 PM

import torch
import torch.nn as nn
from torch.utils import data

import abc
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
import random


class DataSet(metaclass=abc.ABCMeta, data.Dataset):
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


class Net(nn.Module):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass


class TrainMethod:
    @staticmethod
    def batch(train_set, net, loss, lr, epochs):
        losses = []
        X = train_set.features
        Y = train_set.labels
        for epoch in range(epochs):
            Y_hat = net(X).reshape(Y.shape)
            l = loss(Y, Y_hat)
            losses.append(l.detach())
            l.backward()
            with torch.no_grad():
                for param in net.parameters():
                    param.data.sub_(lr * param.grad)
                net.zero_grad()
        return losses

    @staticmethod
    def sgd(train_set, net, loss, lr):
        losses = []
        indies = list(range(len(train_set)))
        random.shuffle(indies)
        for i in indies:
            X, Y = train_set[i]
            Y_hat = net(X).reshape(Y.shape)
            l = loss(Y, Y_hat)
            if i % 10 == 0:
                losses.append(l.detach())
            l.backward()
            with torch.no_grad():
                for param in net.parameters():
                    param.data.sub_(lr * param.grad)
                net.zero_grad()
        return losses

    @staticmethod
    def mini_batch(train_set, net, loss, lr, epochs, batch_size):
        losses = []
        sample_num = len(train_set)
        ind = list(range(sample_num))
        for epoch in range(epochs):
            random.shuffle(ind)
            for i in range(0, sample_num, batch_size):
                X, Y = train_set[i:min(i+batch_size, sample_num)]
                Y_hat = net(X).reshape(Y.shape)
                l = loss(Y, Y_hat)
                losses.append(l.detach())
                l.backward()
                with torch.no_grad():
                    for param in net.parameters():
                        param.data.sub_(lr * param.grad)
                    net.zero_grad()


if __name__ == '__main__':
    pass
