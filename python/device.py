# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Time  :  2022-08-03


def to_device(x, device, device_num=0):
    if device == "cpu":
        return x.cpu()
    elif device == "cuda" or device == "gpu":
        return x.cuda(device_num)


if __name__ == '__main__':
    pass
