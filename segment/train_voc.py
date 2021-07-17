from tqdm import tqdm

from voc_data import VOC2012SegDataIter, VOC_COLORMAP
import torch
from torch import nn, optim
import numpy as np
from torchvision import models, transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from models import ResNet, FCN, VGGNet
from vit import SETR
import random as rm
from utils import ComputeIoU


# 为重绘图像时索引方便起见，将 VOC_COLORMAP 转换为 numpy 矩阵
VOC_COLORMAP_NP = np.array(VOC_COLORMAP, dtype=np.uint8)

# 声明设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 训练，一个 epoch
def train():
    train_loss, train_acc = 0, 0
    batch_count = 0

    # 定义 IoU 算子，对全部训练数据计算 mIoU
    compute_iou = ComputeIoU(num_classes=num_classes)
    miou = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        preds = net(imgs)  # [B, C, H, W]

        # 转置为 [B, H, W, C]，再变形为 [B * H * W, C]
        preds = preds.permute([0, 2, 3, 1]).reshape([-1, num_classes])
        labels = labels.permute([0, 2, 3, 1]).reshape([-1, num_classes])
        # print(preds.size(), labels.size())

        # 计算损失
        loss = lossFN(preds, torch.argmax(labels, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 计算准确率
        # labels/preds: [B, C, H, W]
        labels_map = torch.argmax(labels, dim=1)  # [B, H, W] 各元素的值在 0~C 上，[i, j, k] 给出了第 i 个 img 像素 (j,k) 的类别
        preds_map = torch.argmax(preds, dim=1)  # [B, H, W] 各元素的值在 0~C 上，[i, j, k] 给出了第 i 个 img 像素 (j,k) 的类别
        train_acc += torch.mean(torch.eq(labels_map, preds_map).float()).item()

        # 计算 mIoU
        compute_iou(preds_map, labels_map)  # 这里做的其实只是一个累加混淆矩阵的步骤

        batch_count += 1

    train_loss /= batch_count
    train_acc /= batch_count
    miou = compute_iou.get_miou(ignore=0)

    print("Epoch %3d, loss=%.4f, acc=%.2f%%, miou=%.2f%%" % (epoch + 1, train_loss, train_acc * 100, 100 * miou))

    # 每一轮训练完成后测试一次
    val_loss, val_acc = valid()

    return (train_loss, val_loss), (train_acc, val_acc)


# 验证
def valid():
    net.eval()
    cost, acc = 0, 0

    with torch.no_grad():
        n = rm.randint(0, len(val_loader) - 1)  # 从所有验证数据中随机选择一批数据用于绘图验证

        # 定义 IoU 算子，对全部验证数据计算 mIoU
        compute_iou = ComputeIoU(num_classes=num_classes)
        miou = 0

        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = net(imgs)
            # outputs = torch.sigmoid(outputs)  # output.shape is torch.Size([4,2,160,160])

            labels_map = torch.argmax(labels, dim=1)  # [B, H, W] 各元素的值在 0~C 上，[i, j, k] 给出了第 i 个 img 像素 (j,k) 的类别
            preds_map = torch.argmax(outputs, dim=1)  # [B, H, W] 各元素的值在 0~C 上，[i, j, k] 给出了第 i 个 img 像素 (j,k) 的类别
            # 计算准确率
            acc += torch.mean(torch.eq(labels_map, preds_map).float()).item()
            # 计算 mIoU
            compute_iou(preds_map, labels_map)  # 这里做的其实只是一个累加混淆矩阵的步骤

            # 转置为 [B, H, W, C]，再变形为 [B * H * W, C]
            outputs = outputs.permute([0, 2, 3, 1]).reshape([-1, num_classes])
            labels = labels.permute([0, 2, 3, 1]).reshape([-1, num_classes])

            loss = lossFN(outputs, torch.argmax(labels, dim=1))
            cost += loss.item()

            # 绘图部分，直接观察准确率情况
            if epoch % 10 == 0 and i == n:
                b = rm.randint(0, batch_size - 1)  # 从一批验证数据中随机选择一组数据用于绘图验证
                img = transforms.ToPILImage()(make_grid(imgs[b].cpu(), normalize=True))
                label_map = labels_map[b].cpu().numpy()  # [H, W] 各元素的值在 0~C 上，给出了 img 每个像素的类别
                pred_map = preds_map[b].cpu().numpy()  # [H, W] 各元素的值在 0~C 上，给出了 img 每个像素的类别
                label_img = VOC_COLORMAP_NP[label_map]
                pred_img = VOC_COLORMAP_NP[pred_map]

                plt.subplot(2, 2, 1)
                plt.imshow(img)
                plt.title('img')
                plt.subplot(2, 2, 3)
                plt.imshow(label_img)
                plt.title('label')
                plt.subplot(2, 2, 4)
                plt.imshow(pred_img)
                plt.title('predict')
                plt.pause(0.5)

        cost /= len(val_loader)
        acc /= len(val_loader)
        miou = compute_iou.get_miou(ignore=0)

        print('\tValid Result: loss=%.4f, acc=%.2f%%, miou=%.2f%%' % (cost, 100 * acc, 100 * miou))

    return cost, acc


if __name__ == '__main__':
    num_classes = 21
    batch_size = 32
    num_epochs = 100  # 100

    vgg_model = VGGNet(requires_grad=True, show_params=False)
    # use FCN(VGG)
    # net = FCN(pretrained_net=vgg_model, n_class=num_classes).to(device)
    # use ResNet + FCN/ASPP
    net = ResNet(num_classes, use_aspp=True).to(device)
    # use ViT(SETR)
    # net = model = SETR(
    #     image_size=480,
    #     patch_size=16,
    #     num_classes=num_classes,
    #     emb_dim=128,
    #     depth=24,  # 按照原文，使用 24 层的 Transformer
    #     heads=8,
    #     mlp_dim=128
    # ).to(device)

    # 加载数据集
    train_loader, val_loader = VOC2012SegDataIter(batch_size=batch_size, size=480)

    # 损失函数
    # 用 BCE 的话不用给 preds 和 labels 做变形，就是对 preds 中的每一个元素做二分类，但是对于多分类问题交叉熵感觉上更可取，
    # 个人认为如果是之区别前景后景的二分类的话可以用 BCE
    # lossFN = nn.BCEWithLogitsLoss()  # 二元交叉熵，自带 sigmoid 效果
    lossFN = nn.CrossEntropyLoss()  # 交叉熵，自带 softmax 效果

    # 优化器
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.7, weight_decay=1e-5)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.7, weight_decay=1e-5)  # ViT

    # 记录器
    losses, accs = [], []

    for epoch in range(num_epochs):
        loss, acc = train()
        losses.append(loss)
        accs.append(acc)

    plt.figure()
    plt.plot(range(len(losses)), [l[0] for l in losses], label='train')
    plt.plot(range(len(losses)), [l[1] for l in losses], label='val')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss curve')
    plt.show()

    plt.figure()
    plt.plot(range(len(accs)), [a[0] for a in accs], label='train')
    plt.plot(range(len(accs)), [a[1] for a in accs], label='val')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.title('Accuracy curve')
    plt.show()
