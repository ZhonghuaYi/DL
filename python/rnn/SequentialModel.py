# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-08-15

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from python.dataset import SequentialData
import python.trainmethod as trainmethod


def data_generate():
    num = 1000
    time = torch.arange(1, num+1, dtype=torch.float32)
    x = torch.sin(0.01*time) + torch.normal(0, 0.02, (num,))
    tau = 4
    features = torch.zeros((num-tau, tau))
    for i in range(tau):
        features[:, i] = x[i:i+num-tau]
    labels = x[tau:].reshape(-1, 1)
    return features, labels


def net_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class Net(nn.Module):
    def __init__(self, tau):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Linear(int(tau), 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 1))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    tau = 5
    seq = SequentialData(1000, tau)
    train_set = seq.slide(0, 600)

    net = Net(tau)
    net.apply(net_init)
    print(f"Num of parameters: {sum(param.numel() for param in net.parameters())}")
    lr = 0.01
    epochs = 100
    batch_size = 100
    device = "cpu"
    device_num = 0
    loss = nn.MSELoss()
    train = trainmethod.mini_batch
    trainer = torch.optim.Adam(net.parameters(), lr, weight_decay=0)

    losses = train(seq, net, loss, trainer, epochs, batch_size, device, device_num)
    # print(losses)
    plt.plot(np.array(range(len(losses))), np.array(losses))
    plt.show()

    with torch.no_grad():
        time = np.arange(0, 1000)
        x = seq.features
        y = net(x)
        plt.plot(time[:1000-tau], x, c='b')
        plt.plot(time[tau:], y, c='r')
        plt.show()
