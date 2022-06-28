# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  6/28/2022
# @Time  :  3:58 PM

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


class SquareLossDecent(GradDecent):
    @staticmethod
    def decent(model, lr, train_X, train_Y, train_Y_hat):
        m = len(train_Y_hat)
        w_grad = np.sum((train_Y_hat - train_Y) * train_X, axis=0) / m
        b_grad = np.sum((train_Y_hat - train_Y), axis=0) / m
        model.w_hat = model.w_hat - lr * w_grad
        model.b_hat = model.b_hat - lr * b_grad


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
    linear_model = LinearModel([2.1, 1.4], 0.5)
    linear_model.data_generate(1000)
    linear_model.data_scale()
    linear_model.train(0.1, 1000, SquareLoss.loss, SquareLossDecent.decent, LinearTrain.batch_train)
    linear_model.data_plot(flag="3d")
    plt.show()
