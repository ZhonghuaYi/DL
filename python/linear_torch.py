# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/4/2022
# @Time  :  10:09 PM

import torch
import torch.nn as nn
from torch.utils import data
import random


class LinearData(data.Dataset):
    def __init__(self, sample_num, feature_num, w, b):
        features = torch.normal(2, 1.4, (sample_num, feature_num))
        labels = features @ w + b
        labels += torch.normal(0, 0.5, labels.shape)
        self.features = features
        self.labels = labels

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


if __name__ == '__main__':
    pass
