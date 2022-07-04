# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/4/2022
# @Time  :  10:15 PM

import torch
import torch.nn as nn
from torch.utils import data

import abc


class DataSet(metaclass=abc.ABCMeta):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def data_iter(self):
        pass


if __name__ == '__main__':
    pass
