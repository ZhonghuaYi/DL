# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/2/2022
# @Time  :  2:42 PM
import numpy as np

from python.template import *


def softmax(X):
    X_exp = np.exp(X)
    print(X_exp)
    row_sum = np.sum(X_exp, axis=1).reshape(-1, 1)
    print(row_sum)
    return X_exp / row_sum


def one_hot(Y):
    m = len(Y)
    new_Y = np.zeros((m, 10))
    for i in range(m):
        new_Y[i, Y[i]] = 1
    return new_Y

class SoftmaxData(DataSet):
    """softmax的数据"""

    def data_scale(self, x_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None:
            x_mean, x_maximum = x_parameter
            self.X = (self.X - x_mean) / x_maximum
            return x_mean, x_maximum

        elif x_parameter is None:
            x_mean = np.mean(self.X, axis=0)
            x_maximum = np.max(self.X, axis=0)
            self.X = (self.X - x_mean) / x_maximum
            return x_mean, x_maximum


class CrossEntropyLoss(Loss):
    @staticmethod
    def loss(Y_hat, Y):
        m = len(Y)
        return - np.sum((Y * np.log(Y_hat)) / m)

    @staticmethod
    def regularized_loss(model, Y_hat, Y, lamda):
        m = len(Y)
        regular_item = lamda / (2 * m) * (np.sum(model.w_hat ** 2) +
                                          np.sum(model.b_hat ** 2))
        return CrossEntropyLoss.loss(Y_hat, Y) + regular_item


class CrossEntropyLossDecent(GradDecent):
    @staticmethod
    def decent(model, lr, X, Y, Y_hat):
        m = len(Y_hat)
        w_grad = np.sum(X * (Y_hat - Y), axis=0) / m
        b_grad = np.sum(Y_hat - Y, axis=0) / m
        model.w_hat = model.w_hat - lr * w_grad
        model.b_hat = model.b_hat - lr * b_grad

    @staticmethod
    def regular_decent(model, lr, lamda, X, Y, Y_hat):
        m = len(Y_hat)
        w_grad = np.sum(X * (Y_hat - Y), axis=0) / m
        b_grad = np.sum(Y_hat - Y, axis=0) / m
        model.w_hat = model.w_hat - lr * (w_grad + lamda / m * w_grad)
        model.b_hat = model.b_hat - lr * (b_grad + lamda / m * b_grad)


class SoftmaxTrain(Train):
    pass


class SoftmaxModel(Model):
    def __init__(self):
        self.train_set = SoftmaxData()
        self.valid_set = SoftmaxData()
        self.test_set = SoftmaxData()
        self.w_hat = np.random.normal(0, 0.1, 2)
        self.b_hat = np.random.normal(0, 0.1, 1)

    def __str__(self):
        return f"w_hat: {self.w_hat} b_hat: {self.b_hat}"

    def data_generate(self):
        from python.data.mnist import load_mnist
        train_images, train_labels, test_images, test_labels = load_mnist("../data/")
        self.train_set = SoftmaxData(train_images, train_labels)
        self.test_set = SoftmaxData(test_images, test_labels)

    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        x_parameter = self.train_set.data_scale()
        for data in (self.valid_set, self.test_set):
            data.data_scale(x_parameter)

    def data_plot(self, flag="2d"):
        """绘制训练集、测试集在一个图中"""
        if flag == "2d":
            plt.scatter(self.train_set.X[:, 0], self.train_set.Y, s=10, c='r')
            plt.scatter(self.test_set.X[:, 0], self.test_set.Y, s=10, c='y')
        elif flag == "3d":
            fig = plt.figure("3d data")
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(self.train_set.X[:, 0],
                         self.train_set.X[:, 1],
                         self.train_set.Y,
                         s=10,
                         c='r')
            ax.scatter(self.test_set.X[:, 0],
                         self.test_set.X[:, 1],
                         self.test_set.Y,
                         s=10,
                         c='y')
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Label")

    def predict(self, X):
        """根据X预测Y"""
        # print(X.shape, self.w_hat.shape)
        Y_hat = softmax(np.matmul(X, self.w_hat) + self.b_hat)
        return Y_hat

    def accuracy(self, label="train_set"):
        """计算正确率"""
        data = self.__dict__[label]
        result = (self.predict(data.X) > 0.5).astype(data.Y.dtype)
        accuracy_rate = np.sum((result == data.Y).astype(np.uint8)) / len(data.Y)
        print(f"Accuracy of {label}: {accuracy_rate}")

    def precision(self, label="train_set"):
        """计算查全率"""
        data = self.__dict__[label]
        result = (self.predict(data.X) > 0.5).astype(data.Y.dtype)
        result_1 = result * data.Y
        precision_rate = np.sum(result_1) / np.sum(data.Y)
        print(f"Precision of {label}: {precision_rate}")

    def recall(self, label="train_set"):
        """计算查全率"""
        data = self.__dict__[label]
        result = (self.predict(data.X) > 0.5).astype(data.Y.dtype)
        result_1 = result * data.Y
        recall_rate = np.sum(result_1) / np.sum(result)
        print(f"Precision of {label}: {recall_rate}")


if __name__ == '__main__':
    # 设置训练的参数
    train_dict = {
        "lr": 0.1,
        "epochs": 1000,
        "loss_func": CrossEntropyLoss.loss,
        "decent": CrossEntropyLossDecent.decent,
        "regularized_loss_func": CrossEntropyLoss.regularized_loss,
        "regular_decent": CrossEntropyLossDecent.regular_decent,
        "lamda": 2,
        "training": SoftmaxTrain.batch_train,
    }
    train_parameter = TrainParameter()
    train_parameter.set(**train_dict)

    m = SoftmaxModel()  # 生成模型实例
    m.data_generate()  # 生成模型的数据
    print(m.train_set.Y.shape)
    b = one_hot(m.train_set.Y)
    print(len(b))
