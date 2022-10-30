import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





# 通常批标准化放在全连接层的后面 激活函数前面
class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1), nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2), nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def Drawpicture(epoch,Accuracy_list,Loss_list):
    # 绘画损失函数图像和准确率图像
    x1 = range(0, epoch + 1)
    x2 = range(0, epoch + 1)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")





# 数据集
train_dataset = dsets.MNIST(root='data/',  # 选择数据的根目录
                            train=True,  # 选择训练集
                            transform=transforms.ToTensor(),  # 转换成tensor变量
                            download=True)  # 不从网络上download图片
test_dataset = dsets.MNIST(root='data/',  # 选择数据的根目录
                           train=False,  # 选择训练集
                           transform=transforms.ToTensor(),  # 转换成tensor变量
                           download=True)  # 不从网络上download图片
# 加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,  # 每一次训练选用的数据个数
                                           shuffle=False)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,  # 每一次训练选用的数据个数
                                          shuffle=False)
# 定义学习率，训练次数，损失函数，优化器
learning_rate = 0.01
epoch = 10
criterion = nn.CrossEntropyLoss()
model = Batch_Net(28 * 28, 300, 100, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 模型进行训练
Loss_list=[]#存储损失函数的值
Accuracy_list=[]
for epoch in range(epoch):
    train_loss = 0
    train_acc = 0
    for img, label in train_loader:
        #print('img:{}'.format(img.size()))
        img = Variable(img.view(img.size(0), -1))
        #print('newimg:{}'.format((img.size())))
        label = Variable(label)
        #print('label:{}'.format((label.size())))
        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        # print('output.max:{},output.max(1):{}'.format((output.max), output.max(1)))
        _, pred = output.max(1)
        #torch.max(0)和 torch.max(1)分别是找出tensor里每列/每行中最大的值，并返回索引（即为对应的预测数字）,返回两个数字
        num_correct = (pred == label).sum().item() # 如果预测结果和真实值相等则计数 +1
        acc = num_correct / img.shape[0]
        train_acc += acc

    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, train_loss / len(train_loader),
                                                                    train_acc / len(train_loader)))
    Loss_list.append(train_loss / len(train_loader))
    Accuracy_list.append(train_acc / len(train_loader))
Drawpicture(epoch,Loss_list,Accuracy_list)



# 测试网络模型
model.eval()
eval_loss = 0
eval_acc = 0
for img, label in test_loader:
    #print(img.shape)
    img = Variable(img.view(img.size(0), -1))#转换img格式，使之符合输入的神经元数目
    #print(img.shape)
    label = Variable(label)
    output = model(img)#model的输入是一个[*,28*28]的Tensor,获取的img要先转换成这个格式
    loss = criterion(output, label)
    eval_loss += loss.data * img.size(0)
    _, pred = torch.max(output, 1)
    num_correct = (pred == label).sum().item()
    eval_acc += num_correct
print("Test Loss:{:.6f},Acc:{:.6f}".format(eval_loss / 10000, eval_acc / 10000))

torch.save(model, 'model.tar')  # 保存模型
