# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/5/2022
# @Time  :  7:57 PM

from python.template_torch import *
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    if type(x) == torch.Tensor:
        return 1 / (1 + torch.exp(-x))
    elif type(x) == np.ndarray:
        return 1 / (1 + np.exp(-x))


class CrossEntropyLoss:
    @staticmethod
    def loss(Y_hat, Y):
        m = Y.shape[0]
        return - torch.sum((Y * torch.log(Y_hat) + (1 - Y) * torch.log(1 - Y_hat)) / m)

    @staticmethod
    def regularized_loss(Y_hat, Y, w_hat, b_hat, lamda):
        m = Y.shape[0]
        regular_item = lamda / (2 * m) * (torch.sum(w_hat ** 2, dim=1) +
                                          torch.sum(b_hat ** 2, dim=1))
        return CrossEntropyLoss.loss(Y_hat, Y) + regular_item


class LogisticData(DataSet):
    def __init__(self, num1, num2):
        X11 = np.random.normal(-2, 1, (num1, 1))
        X12 = np.random.normal(2, 1, (num1, 1))
        Y1 = np.zeros((num1, 1))
        X21 = np.random.normal(2, 1, (num2, 1))
        X22 = np.random.normal(-2, 1, (num2, 1))
        Y2 = np.ones((num2, 1))
        Z = np.hstack((np.vstack((X11, X21)), np.vstack((X12, X22)), np.vstack((Y1, Y2))))
        np.random.shuffle(Z)
        X = Z[:, 0:-1]
        Y = Z[:, -1].reshape(-1, 1)
        self.features = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(Y).float()

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
            x_mean, x_maximum = x_parameter
            self.features = (self.features - x_mean) / x_maximum
            return x_mean, x_maximum

        elif x_parameter is None:
            x_mean = self.features.mean(dim=0)
            x_maximum = self.features.max(dim=0).values
            self.features = (self.features - x_mean) / x_maximum
            return x_mean, x_maximum


class LogisticNet(Net):
    def __init__(self, feature_num, label_num):
        super().__init__()
        self.fc1 = nn.Linear(feature_num, label_num)

    def forward(self, X):
        Y_hat = sigmoid(self.fc1(X))
        return Y_hat


def accuracy(data, net):
    """计算正确率"""
    result = (net(data.features) > 0.5).clone().detach().reshape(data.labels.shape).int()
    accuracy_rate = (result == data.labels).sum() / data.labels.shape[0]
    print(f"Accuracy: {accuracy_rate}")


def precision(data, net):
    """计算查准率"""
    result = (net(data.features) > 0.5).clone().detach().reshape(data.labels.shape).int()
    result_1 = result * data.labels
    precision_rate = result_1.sum() / data.labels.sum()
    print(f"Precision: {precision_rate}")


def recall(data, net):
    """计算查全率"""
    result = (net(data.features) > 0.5).clone().detach().reshape(data.labels.shape).int()
    result_1 = result * data.labels
    recall_rate = result_1.sum() / result.sum()
    print(f"Recall: {recall_rate}")


if __name__ == '__main__':
    train_set = LogisticData(300, 300)
    test_set = LogisticData(200, 200)

    lr = 0.1
    epochs = 10
    batch_size = 50
    net = LogisticNet(2, 1)
    loss = CrossEntropyLoss.loss
    train = TrainMethod.mini_batch
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)

    losses = train(train_set, net, loss, trainer, epochs, batch_size)
    plt.plot(np.array(range(len(losses))), np.array(losses))
    plt.show()

    with torch.no_grad():
        # 画出数据
        train_set.data_plot("3d")
        test_set.data_plot("3d", c='y')
        # 画出用于预测的直线（平面）
        x = np.linspace(-3, 3, 100)
        y = x
        x, y = np.meshgrid(x, y)
        w_hat, b_hat = list(net.fc1.parameters())
        print(f"w_hat: {w_hat.numpy()}")
        print(f"b_hat: {b_hat.numpy()}")
        z = w_hat[0, 0] * x + w_hat[0, 1] * y + b_hat
        fig = plt.figure("3d data")
        ax = fig.axes[0]
        # ax.plot_surface(x, y, z, color=(0, 1, 0, 0.3))
        # 画出直线（平面）经过sigmoid拟合后预测的概率
        ax.plot_surface(x, y, sigmoid(z), color=(0, 0, 1, 0.3))
        ax.set_zlim(-0.5, 1.5)
        plt.show()
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
