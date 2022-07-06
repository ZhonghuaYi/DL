# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/6/2022
# @Time  :  11:00 PM
import torch

from python.template_torch import *
import numpy as np
import matplotlib.pyplot as plt


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


def one_hot(Y):
    y = None
    if type(Y) == torch.Tensor:
        y = Y.numpy()
    m = len(y)
    new_y = np.zeros((m, 10), dtype=np.int32)
    for i in range(m):
        new_y[i, int(y[i])] = 1
    if type(Y) == torch.Tensor:
        return torch.from_numpy(new_y)
    else:
        return new_y



if __name__ == '__main__':
    pass
