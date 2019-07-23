import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
# MNIST数据集已经集成在pytorch datasets中，可以直接调用
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)

        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)#（in_features, out_features）

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))

        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x)  #64*10


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()

        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):

        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return F.log_softmax(x)  #64*10

model = Net0()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):#batch_idx是enumerate（）函数自带的索引，从0开始
        # data.size():[64, 1, 28, 28]
        # target.size():[64]
        output = model(data)
        #output:64*10
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()   # 所有参数的梯度清零
        loss.backward()         #即反向传播求梯度
        optimizer.step()        #调用optimizer进行梯度下降更新参数
    print('loss train {}'.format(loss.data))
    tloss.append(loss.data)
    

def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader)
    print('loss test {}'.format(test_loss.data))
    print('acc test {}'.format(correct))


    eloss.append(test_loss.data)
    eacc.append(correct/10000)


tloss = []
eloss = []
eacc = []

for epoch in range(20):
    train(epoch)
    test()

from matplotlib import pyplot as plt

plt.plot(tloss)
plt.plot(eloss)
plt.title('loss')
plt.show()

plt.plot(eacc)
plt.title('accuracy')
plt.show()
