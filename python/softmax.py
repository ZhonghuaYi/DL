# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/2/2022
# @Time  :  2:42 PM
import numpy as np

from python.template import *


def softmax(X):
    X_exp = np.exp(X)
    row_sum = 0
    if X.ndim == 1:
        row_sum = np.sum(X_exp)
    if X.ndim == 2:
        row_sum = np.sum(X_exp, axis=1).reshape(-1, 1)
    return X_exp / row_sum


def one_hot(Y):
    m = len(Y)
    new_Y = np.zeros((m, 10), dtype=np.int32)
    for i in range(m):
        new_Y[i, int(Y[i])] = 1
    return new_Y


class SoftmaxData(DataSet):
    """softmax的数据"""

    def data_scale(self, x_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None:
            x_mean, x_maximum = x_parameter
            self.X = np.divide(self.X, x_maximum, out=np.zeros_like(self.X, dtype=np.float64),
                               where=x_maximum!=0)
            return x_mean, x_maximum

        elif x_parameter is None:
            x_mean = np.mean(self.X, axis=0)
            x_maximum = np.max(self.X, axis=0)
            self.X = np.divide(self.X, x_maximum, out=np.zeros_like(self.X, dtype=np.float64),
                               where=x_maximum!=0)
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
        w_grad = np.matmul(X.T, Y_hat - Y) / m
        b_grad = np.sum(Y_hat - Y, axis=0) / m
        model.w_hat = model.w_hat - lr * w_grad
        model.b_hat = model.b_hat - lr * b_grad

    @staticmethod
    def regular_decent(model, lr, lamda, X, Y, Y_hat):
        m = len(Y_hat)
        w_grad = np.matmul(X.T, Y_hat - Y) / m
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
        self.w_hat = np.random.normal(1, 0.1, (784, 10))
        self.b_hat = np.random.normal(1, 0.1, 10)

    def __str__(self):
        return f"w_hat: {self.w_hat} b_hat: {self.b_hat}"

    def data_generate(self):
        from python.data.mnist import load_mnist
        train_images, train_labels, test_images, test_labels = load_mnist("./data/")
        train_labels = one_hot(train_labels)
        test_labels = one_hot(test_labels)
        self.train_set = SoftmaxData(train_images, train_labels)
        self.test_set = SoftmaxData(test_images, test_labels)

    def data_scale(self):
        """将数据以训练集的平均值和最大值归一化"""
        x_parameter = self.train_set.data_scale()
        self.test_set.data_scale(x_parameter)

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
        a = np.matmul(X, self.w_hat)
        Y_hat = softmax(np.matmul(X, self.w_hat) + self.b_hat)
        return Y_hat

    def accuracy(self, label="train_set"):
        """计算正确率"""
        data = self.__dict__[label]
        result = one_hot(self.predict(data.X).argmax(axis=1))
        accuracy_rate = np.sum((result*data.Y).astype(np.uint8)) / len(data.Y)
        print(f"Accuracy of {label}: {accuracy_rate}")

    def precision(self, label="train_set"):
        """计算查准率"""
        data = self.__dict__[label]
        result = one_hot(self.predict(data.X).argmax(axis=1))
        result_1 = result * data.Y
        precision_rate = np.sum(result_1, axis=0) / np.sum(data.Y, axis=0)
        print(f"Precision of {label}: {precision_rate}")

    def recall(self, label="train_set"):
        """计算查全率"""
        data = self.__dict__[label]
        result = one_hot(self.predict(data.X).argmax(axis=1))
        result_1 = result * data.Y
        recall_rate = np.sum(result_1, axis=0) / np.sum(result, axis=0)
        print(f"Recall of {label}: {recall_rate}")


if __name__ == '__main__':
    # 设置训练的参数
    train_dict = {
        "lr": 0.1,
        "epochs": 10,
        "batch_size": 50,
        "loss_func": CrossEntropyLoss.loss,
        "decent": CrossEntropyLossDecent.decent,
        "regularized_loss_func": CrossEntropyLoss.regularized_loss,
        "regular_decent": CrossEntropyLossDecent.regular_decent,
        "lamda": 0.1,
        "training": SoftmaxTrain.mini_batch,
    }
    train_parameter = TrainParameter()
    train_parameter.set(**train_dict)

    m = SoftmaxModel()  # 生成模型实例
    m.data_generate()  # 生成模型的数据
    m.data_scale()

    m.train(train_parameter)  # 训练参数
    m.test(train_parameter.loss_func)  # 计算测试集的loss

    # 计算训练集、验证集和测试集的正确率、查准率和查全率
    for data_set in ("train_set", "test_set"):
        print(f"{data_set}:")
        m.accuracy(data_set)
        m.precision(data_set)
        m.recall(data_set)
