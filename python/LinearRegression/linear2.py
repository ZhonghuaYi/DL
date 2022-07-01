# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  6/28/2022
# @Time  :  3:58 PM
import numpy as np

from python.template import *


class LinearData(DataSet):

    def data_scale(self, x_parameter=None, y_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None and y_parameter is not None:
            x_mean, x_maximum = x_parameter
            y_mean, y_maximum = y_parameter
            self.X = (self.X - x_mean) / x_maximum
            self.Y = (self.Y - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)

        elif x_parameter is None and y_parameter is None:
            x_mean = np.mean(self.X, axis=0)
            x_maximum = np.max(self.X, axis=0)
            self.X = (self.X - x_mean) / x_maximum
            y_mean = np.mean(self.Y, axis=0)
            y_maximum = np.max(self.Y, axis=0)
            self.Y = (self.Y - y_mean) / y_maximum
            return (x_mean, x_maximum), (y_mean, y_maximum)


class SquareLoss(Loss):
    @staticmethod
    def loss(Y_hat, Y):
        m = len(Y)
        return np.sum((Y_hat - Y) ** 2. / (2. * m))

    @staticmethod
    def regularized_loss(model, Y_hat, Y, lamda):
        m = len(Y)
        regular_item = lamda / (2 * m) * (np.sum(model.w_hat ** 2, axis=0) +
                                          np.sum(model.b_hat ** 2, axis=0))
        return SquareLoss.loss(Y_hat, Y) + regular_item


class SquareLossDecent(GradDecent):
    @staticmethod
    def decent(model, lr, train_X, train_Y, train_Y_hat):
        m = len(train_Y_hat)
        w_grad = np.sum((train_Y_hat - train_Y) * train_X, axis=0) / m
        b_grad = np.sum((train_Y_hat - train_Y), axis=0) / m
        model.w_hat = model.w_hat - lr * w_grad
        model.b_hat = model.b_hat - lr * b_grad

    @staticmethod
    def regular_decent(model, lr, lamda, train_X, train_Y, train_Y_hat):
        m = len(train_Y_hat)
        w_grad = np.sum((train_Y_hat - train_Y) * train_X, axis=0) / m
        b_grad = np.sum((train_Y_hat - train_Y), axis=0) / m
        model.w_hat = model.w_hat - lr * (w_grad + lamda / m * w_grad)
        model.b_hat = model.b_hat - lr * (b_grad + lamda / m * b_grad)


class LinearTrain(Train):
    pass


class LinearModel(Model):
    def __init__(self, true_w, true_b):
        self.train_set = LinearData()
        self.valid_set = LinearData()
        self.test_set = LinearData()
        self.w = np.asarray(true_w)
        self.b = np.asarray(true_b)
        self.w_hat = np.random.normal(0, 0.1, self.w.shape)
        self.b_hat = np.random.normal(0, 0.1, self.b.shape)

    def __str__(self):
        return f"w:{self.w} b:{self.b}\n" \
               f"w_hat{self.w_hat} b_hat{self.b_hat}\n"

    def data_generate(self, num):
        """产生数据"""
        X = np.random.normal(2, 1.4, (num, len(self.w)))
        Y = np.matmul(X, self.w) + self.b
        Y += np.random.normal(0, 0.5, Y.shape)
        Y = Y.reshape((-1, 1))
        train = (X[0:int(num * 0.6), ...], Y[0:int(num * 0.6), :])
        valid = (X[int(num * 0.6):int(num * 0.8), ...], Y[int(num * 0.6):int(num * 0.8), :])
        test = (X[int(num * 0.8):num, ...], Y[int(num * 0.8):num, :])
        self.train_set = LinearData(train[0], train[1])
        self.valid_set = LinearData(valid[0], valid[1])
        self.test_set = LinearData(test[0], test[1])

    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        x_parameter, y_parameter = self.train_set.data_scale()
        for data in (self.valid_set, self.test_set):
            data.data_scale(x_parameter, y_parameter)
        w = self.w
        b = self.b
        self.w = x_parameter[1] * w.T / y_parameter[1]
        self.b = (np.sum(x_parameter[0] * w.T) + b - y_parameter[0]) / y_parameter[1]

    def data_plot(self, flag="2d"):
        super().data_plot(flag)

    def predict(self, X):
        """根据X预测Y"""
        Y_hat = np.matmul(X, self.w_hat) + self.b_hat
        return Y_hat.reshape((-1, 1))


if __name__ == '__main__':
    train_dict = {
        "lr": 0.1,
        "epochs": 1000,
        "loss_func": SquareLoss.loss,
        "decent": SquareLossDecent.decent,
        "regularized_loss_func": SquareLoss.regularized_loss,
        "regular_decent": SquareLossDecent.regular_decent,
        "lamda": 5,
        "training": LinearTrain.batch_train,
    }
    train_parameter = TrainParameter()
    train_parameter.set(**train_dict)
    linear_model = LinearModel([2.1, 1.4], 0.5)
    linear_model.data_generate(1000)
    linear_model.data_scale()
    linear_model.train(train_parameter)
    linear_model.valid(train_parameter.loss_func)
    linear_model.test(train_parameter.loss_func)
    linear_model.data_plot(flag="3d")
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)
    z = linear_model.w_hat[0] * x + linear_model.w_hat[1] * y + linear_model.b_hat
    fig = plt.figure("3d data")
    ax = fig.axes[0]
    ax.plot_surface(x, y, z, color=(0, 1, 0, 0.3))
    plt.show()
