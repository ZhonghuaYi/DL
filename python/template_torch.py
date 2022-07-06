# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/4/2022
# @Time  :  10:15 PM

import abc
import torch
import torch.nn as nn
from torch.utils import data
import random


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


class Net(nn.Module):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass


class TrainMethod:
    @staticmethod
    def batch(train_set, net, loss, trainer, epochs):
        losses = []
        X = train_set.features
        Y = train_set.labels
        for epoch in range(epochs):
            trainer.zero_grad()
            Y_hat = net(X).reshape(Y.shape)
            l = loss(Y_hat, Y)
            losses.append(l.detach())
            l.backward()
            trainer.step()
        return losses

    @staticmethod
    def sgd(train_set, net, loss, trainer):
        losses = []
        indies = list(range(len(train_set)))
        random.shuffle(indies)
        for i in indies:
            trainer.zero_grad()
            X, Y = train_set[i]
            Y_hat = net(X).reshape(Y.shape)
            l = loss(Y_hat, Y)
            if i % 10 == 0:
                losses.append(l.detach())
            l.backward()
            trainer.step()
        return losses

    @staticmethod
    def mini_batch(train_set, net, loss, trainer, epochs, batch_size):
        losses = []
        sample_num = len(train_set)
        ind = list(range(sample_num))
        for epoch in range(epochs):
            random.shuffle(ind)
            for i in range(0, sample_num, batch_size):
                X, Y = train_set[i:min(i+batch_size, sample_num)]
                trainer.zero_grad()
                Y_hat = net(X).reshape(Y.shape)
                l = loss(Y_hat, Y)
                losses.append(l.detach())
                l.backward()
                trainer.step()
        return losses


if __name__ == '__main__':
    pass
