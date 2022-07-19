# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/13/2022
# @Time  :  12:50 PM

import random

import torch


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


def sgd(train_set, net, loss, trainer):
    losses = []
    indies = list(range(len(train_set)))
    random.shuffle(indies)
    for i in indies:
        trainer.zero_grad()
        X, Y = train_set[i]
        X = X.reshape(1, *X.shape)
        Y = Y.reshape(1, *Y.shape)
        Y_hat = net(X).reshape(Y.shape)
        l = loss(Y_hat, Y)
        if i % 10 == 0:
            losses.append(l.detach())
        l.backward()
        trainer.step()
    return losses


def mini_batch(train_set, net, loss, trainer, epochs, batch_size):
    losses = []
    for epoch in range(epochs):
        for X, Y in train_set.data_iter(batch_size):
            trainer.zero_grad()
            Y_hat = net(X).reshape(Y.shape)
            l = loss(Y_hat, Y)
            losses.append(l.detach())
            l.backward()
            trainer.step()
    return losses