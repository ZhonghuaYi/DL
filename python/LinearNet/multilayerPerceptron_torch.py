# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/8/2022
# @Time  :  1:39 PM

from python.template_torch import *
from softmax_torch import *


class PerceptronNet(Net):
    def __init__(self, feature_num, label_num):
        super(PerceptronNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(feature_num, 256),
                                nn.ReLU(),
                                nn.Linear(256, label_num))

    def forward(self, X):
        Y_hat = self.fc(X)
        return Y_hat


if __name__ == '__main__':
    train_set = MnistData("train")
    test_set = MnistData("test")
    x_mean, x_maximum = train_set.data_scale()
    test_set.data_scale((x_mean, x_maximum))

    net = PerceptronNet(784, 10)
    lr = 0.1
    epochs = 10
    batch_size = 50
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
