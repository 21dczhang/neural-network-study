import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import torch.nn as nn
from PIL import Image
import os


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


# if __name__ =='__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = torch.load('model.tar') #加载模型
#     model = model.to(device)
#     model.eval()    #把模型转为test模式
#     img = cv2.imread("0.bmp")  #读取要预测的图片
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#图片转为灰度图，因为mnist数据集都是灰度图
#     img=np.array(img).astype(np.float32)
#     img=np.expand_dims(img,0)
#     img=np.expand_dims(img,0)#扩展后，为[1，1，28，28]
#     img=torch.from_numpy(img)
#     img = img.to(device)
#     img = Variable(img.view(img.size(0), -1))
#     output=model(img)
#     prob = F.softmax(output, dim=1)
#     prob = Variable(prob)
#     prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
#     print(prob)  #prob是10个分类的概率
#     pred = np.argmax(prob) #选出概率最大的一个
#     print(pred.item())

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((1.1618,), (1.1180,))])
transform = transforms.ToTensor()


def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))]


images = np.array([])
file = get_files('picture')

# 获取图片
for i, item in enumerate(file):
    print('Processing %i of %i (%s)' % (i + 1, len(file), item))
    image = transform(Image.open(item).convert('L'))
    images = np.append(images, image.numpy())

img = images.reshape(-1, 1, 28, 28)
img = torch.from_numpy(img).float()
label = torch.ones(5, 1).long()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.tar')  # 加载模型
model = model.to(device)
model.eval()  # 把模型转为test模式
img = np.array(img).astype(np.float32)
img = np.expand_dims(img, 0)
img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
img = torch.from_numpy(img)
img = img.to(device)
img = Variable(img.view(-1, 784))
print(img.shape)
output = model(img)
prob = F.softmax(output, dim=1)
prob = Variable(prob)
prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
print(prob)  # prob是10个分类的概率
for i in range(prob.shape[0]):
    pred = np.argmax(prob, axis=1)  # 选出概率最大的一个
    print(pred)
