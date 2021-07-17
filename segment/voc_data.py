import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from math import ceil

from os import path

# 这下面应该是给出了 21 种 RGB 组合，分别实现了 21 种不同的颜色，VOC 的标注图片中分别用这 21 种颜色标注出了 21 种不同的物体
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def voc_label_indices(colormap, colormap2label):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def read_voc_images(dir, train=True, max_num=None):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (dir, 'train.txt' if train else 'val.txt')
    # print(txt_fname)

    with open(txt_fname, 'r') as f:
        images = f.read().split()
    # print(len(images))

    if max_num is not None:
        images = images[:min(max_num, len(images))]
    imgs, labels = [None] * len(images), [None] * len(images)

    for i, fname in enumerate(images):
        imgs[i] = Image.open('%s/JPEGImages/%s.jpg' % (dir, fname)).convert("RGB")
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (dir, fname)).convert("RGB")

    return imgs, labels  # PIL image


def voc_rand_crop(feature, label, size):
    """
    Random crop feature (PIL image) and label (PIL image).
    """
    # print(feature.size, label.size)

    # 其实用一个 transforms.RandomCrop() 就可以完成的，像下面这样写是为了把 feature 和 label 裁剪为相同的尺寸
    i, j, h, w = transforms.RandomCrop.get_params(feature, output_size=(size, size))

    feature = transforms.functional.crop(feature, i, j, h, w)
    label = transforms.functional.crop(label, i, j, h, w)

    return feature, label


class VOCSegDataset(Dataset):
    def __init__(self, train, size, data_dir, colormap2label, max_num=None, transform=None):
        """
        crop_size: (h, w)
        """
        # 对数据集中的图片做变换，这就是常规的图片到张量的变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if transform is None else transform

        # 定义裁剪后图片的尺寸，虽然说模型可以输入任意尺寸的图片并输出与其相同形状的特征图，但是训练时还是应当保持数据集中所有图片大小相同
        self.size = size  # (h, w)

        # 读出图片和标注图片
        imgs, labels = read_voc_images(dir=data_dir, train=train, max_num=max_num)

        # 这里的过滤主要是把那些经裁剪后形状仍不合格的图片和标注图片滤去
        self.imgs, self.labels = self.filter(imgs, labels)  # PIL image
        # self.labels = self.filter(labels)  # PIL image
        # self.imgs = imgs  # PIL image
        # self.labels = labels  # PIL image
        self.colormap2label = colormap2label

        print('read ' + str(len(self.imgs)) + ' %s examples' % ('train' if train else 'valid'))

    def filter(self, imgs, tags):
        # print(imgs.size(), labels.size())
        _imgs, _tags = [], []
        for img, tag in zip(imgs, tags):
            assert img.size == tag.size, 'size of img must be equal to size of tag'
            # 首先计算原图片的长宽
            w, h = img.size[0], img.size[1]
            # 计算缩放后图片的长宽
            s = min(w, h)
            scale = self.size / s
            nw = ceil(w * scale)  # 向上取整
            nh = ceil(h * scale)  # 向上取整
            # 对图片和标注图片进行缩放
            # _img = img.resize((nw, nh), Image.ANTIALIAS)
            # print(_img.size)
            _imgs.append(img.resize((nw, nh), Image.ANTIALIAS))
            _tags.append(tag.resize((nw, nh), Image.ANTIALIAS))
        # return [img for img in _imgs if (img.size[0] >= self.size and img.size[1] >= self.size)], \
        #        [tag for tag in _tags if (tag.size[0] >= self.size and tag.size[1] >= self.size)]
        return _imgs, _tags

    def __getitem__(self, idx):
        img, label = voc_rand_crop(self.imgs[idx], self.labels[idx], self.size)  # * 用于解包 list 打包的参数
        label_img = np.array(label).astype(np.uint8)  # (320, 480, 3)
        label_id = voc_label_indices(label, self.colormap2label).numpy().astype('uint8')

        # 统一 GT
        h, w = label_img.shape[0: 2]
        # 最后的标注是 [n_class, h, w] 的一个张量，并且是 uint8，每个元素均为 0 或 1
        # 若 [c, i, j] = 1 这说明原图像 img 的第 (i, j) 个像素属于第 c 类，否则该像素不属于第 c 类
        target = torch.zeros(21, h, w)
        for c in range(21):
            target[c][label_id == c] = 1

        return self.transform(img), target

    def __len__(self):
        return len(self.imgs)


def VOC2012SegDataIter(batch_size=64, size=480, num_workers=4, max_num=None):
    # 这就是一张颜色（三维 RGB 向量）到类别编号（int）的映射表，之所以不用 dict 是为了索引时不用 for 循环，而用矩阵索引加速计算
    colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
    # print(colormap2label.size())

    # 下面这句是关键！！
    for i, colormap in enumerate(VOC_COLORMAP):
        id = (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
        # print(id, i)
        colormap2label[id] = i
    # print()

    # 数据集路径
    base_dir = '/home/duanjiaxin/datasets'
    train_dir = path.join(base_dir, 'VOCdevkit/VOC2012')
    val_dir = path.join(base_dir, 'VOCdevkit/VOC2012')

    # 数据集
    train_set = VOCSegDataset(True, size, train_dir, colormap2label, max_num)
    val_set = VOCSegDataset(False, size, val_dir, colormap2label, max_num)

    # 数据集加载器
    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, drop_last=True, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == '__main__':
    train_iter, val_iter = VOC2012SegDataIter(4, 480)
    for imgs, labels in train_iter:
        print(imgs.size())  # [batch_size, 3, 320, 480]
        print(labels.size())  # [batch_size, 21, 320, 480]
        break
