import os
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# from vit_pytorch.efficient import ViT
from models import ViT

# Training settings
BATCH_SIZE = 128
epochs = 200
lr = 3e-5
gamma = 0.7
seed = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)  # 初始数据归一化防止数据过大

# # 数据集预处理
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])

# 数据集、数据集加载器
trainset = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
testset = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# 这里可以将测试模型由 ViT 换成其他模型
model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=128,
    depth=6,
    heads=8,
    mlp_dim=128
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

loss_list = []
acc_list = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(trainloader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(trainloader)
        epoch_loss += loss / len(trainloader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in testloader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(testloader)
            epoch_val_loss += val_loss / len(testloader)

    loss_list.append([epoch_loss.cpu().data.numpy(), epoch_val_loss.cpu().data.numpy()])
    acc_list.append([epoch_accuracy.cpu().data.numpy(), epoch_val_accuracy.cpu().data.numpy()])

    print(f"Epoch: {epoch+1}"
          f" -loss: {epoch_loss:.4f} -acc: {epoch_accuracy:.4f}"
          f" -val_loss: {epoch_val_loss:.4f} -val_acc: {epoch_val_accuracy:.4f}\n"
    )

# 绘制损失、准确率曲线
plt.subplot(121)
plt.plot(range(len(loss_list)), [l[0] for l in loss_list], label='train_loss')
plt.plot(range(len(loss_list)), [l[1] for l in loss_list], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title('Loss Curve of ViT')
plt.subplot(122)
plt.plot(range(len(acc_list)), [a[0] for a in acc_list], label='train_acc')
plt.plot(range(len(acc_list)), [a[1] for a in acc_list], label='val_acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.title('Acc Curve of ViT')
plt.show()

