import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

train_data = torch.load('mnist_feature_train')
test_data = torch.load('mnist_feature_test')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    print(epoch)
    for batch_idx, (data, target) in enumerate(train_data):
        output = model(data)
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()   # 所有参数的梯度清零
        loss.backward()         #即反向传播求梯度
        optimizer.step()        #调用optimizer进行梯度下降更新参数

    loss_train.append(loss)


def test():
    correct = 0
    for data, target in test_data:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('accuracy on the whole test set {}'.format(correct.numpy() / 10000))
    loss_test.append(loss)


loss_train = []
loss_test = []
for epoch in range(40):
    train(epoch)
    test()

from matplotlib import pyplot as plt
plt.plot(loss_train)
plt.plot(loss_test)
plt.axis()
plt.title('loss')
plt.legend('train','test')
plt.show()