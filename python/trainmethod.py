# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/13/2022
# @Time  :  12:50 PM

import random

from python.device import to_device


def batch(train_set, net, loss, trainer, epochs, device="cpu", device_num=0):
    losses = []
    train_set.to_device(device, device_num)
    X = train_set.features
    Y = train_set.labels
    net = to_device(net, device, device_num)
    for epoch in range(epochs):
        trainer.zero_grad()
        Y_hat = net(X).reshape(Y.shape)
        l = loss(Y_hat, Y)
        losses.append(l.detach())
        l.backward()
        trainer.step()
    net.cpu()
    return losses


def sgd(train_set, net, loss, trainer, device="cpu", device_num=0):
    losses = []
    indies = list(range(len(train_set)))
    random.shuffle(indies)
    train_set.to_device(device, device_num)
    net = to_device(net, device, device_num)
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
    net.cpu()
    return losses


def mini_batch(train_set, net, loss, trainer, epochs, batch_size, device="cpu", device_num=0):
    losses = []
    net = to_device(net, device, device_num)
    for epoch in range(epochs):
        for X, Y in train_set.data_iter(batch_size):
            X = to_device(X, device, device_num)
            Y = to_device(Y, device, device_num)
            trainer.zero_grad()
            Y_hat = net(X).reshape(Y.shape)
            l = loss(Y_hat, Y)
            losses.append(l.detach().cpu())
            l.backward()
            trainer.step()
    net.cpu()
    return losses