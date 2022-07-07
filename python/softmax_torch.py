# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/6/2022
# @Time  :  11:00 PM

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
    y = Y
    if type(Y) == torch.Tensor:
        y = Y.numpy()
    m = y.shape[0]
    new_y = np.zeros((m, 10), dtype=np.int32)
    for i in range(m):
        new_y[i, int(y[i])] = 1
    if type(Y) == torch.Tensor:
        return torch.from_numpy(new_y)
    else:
        return new_y


class MnistData(DataSet):
    def __init__(self, label="train"):
        from python.data.mnist import load_mnist
        train_images, train_labels, test_images, test_labels = load_mnist("./data/")
        if label == "train":
            self.features = torch.from_numpy(train_images[:6000, ...]).float()
            self.labels = torch.from_numpy(one_hot(train_labels[:6000, ...])).float()
        elif label == "test":
            self.features = torch.from_numpy(test_images[:1000, ...]).float()
            self.labels = torch.from_numpy(one_hot(test_labels[:1000, ...])).float()


    def data_plot(self, flag="2d", c='r'):
        """绘制数据点图，可以是3d"""
        if flag == "2d":
            plt.scatter(self.features[:, 0], self.labels, s=10, c=c)
        elif flag == "3d":
            fig = plt.figure("3d data")
            if len(fig.axes) == 0:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.axes[0]
            ax.scatter(self.features[:, 0], self.features[:, 1], self.labels, s=10, c=c)

    def data_scale(self, x_parameter=None):
        """将数据以其平均值和最大值归一化至(-1,1)"""
        if x_parameter is not None:
            mask = self.features == 0
            x_mean, x_maximum = x_parameter
            self.features = (self.features - x_mean) / x_maximum
            self.features[mask] = 0
            return x_mean, x_maximum

        elif x_parameter is None:
            mask = self.features == 0
            x_mean = self.features.mean(dim=0)
            x_maximum = self.features.max(dim=0).values
            self.features = (self.features - x_mean) / x_maximum
            self.features[mask] = 0
            return x_mean, x_maximum


class SoftmaxNet(Net):
    def __init__(self, feature_num, label_num):
        super().__init__()
        self.fc1 = nn.Linear(feature_num, label_num)

    def forward(self, X):
        Y_hat = self.fc1(X)
        return Y_hat


def accuracy(data, net):
    """计算正确率"""
    result = one_hot(net(data.features).argmax(dim=1))
    accuracy_rate = (result * data.labels).sum() / data.labels.shape[0]
    print(f"Accuracy: {accuracy_rate}")


def precision(data, net):
    """计算查准率"""
    result = one_hot(net(data.features).argmax(dim=1))
    result_1 = result * data.labels
    precision_rate = result_1.sum(dim=0) / data.labels.sum(dim=0)
    print(f"Precision: {precision_rate}")


def recall(data, net):
    """计算查全率"""
    result = one_hot(net(data.features).argmax(dim=1))
    result_1 = result * data.labels
    recall_rate = result_1.sum(dim=0) / result.sum(dim=0)
    print(f"Recall: {recall_rate}")


if __name__ == '__main__':
    train_set = MnistData("train")
    test_set = MnistData("test")
    x_mean, x_maximum = train_set.data_scale()
    test_set.data_scale((x_mean, x_maximum))

    lr = 0.1
    epochs = 10
    batch_size = 50
    net = SoftmaxNet(784, 10)
    loss = nn.CrossEntropyLoss()
    train = TrainMethod.mini_batch
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)

    losses = train(train_set, net, loss, trainer, epochs, batch_size)
    plt.plot(np.array(range(len(losses))), np.array(losses))
    plt.show()

    with torch.no_grad():
        # 训练集的正确率、查准率和查全率
        print("\nTrain set:")
        accuracy(train_set, net)
        precision(train_set, net)
        recall(train_set, net)
        # 测试集的正确率、查准率和查全率
        print("\nTest set:")
        accuracy(test_set, net)
        precision(test_set, net)
        recall(test_set, net)
