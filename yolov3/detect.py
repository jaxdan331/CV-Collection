import argparse
import time
from sys import platform

import torch
import numpy as np

from model import *
from utils.datasets import *
from utils.utils import *

import os
import random as rm

"""
对单个图片进行预测，并绘制结果
"""

def detect(
        cfg,
        data_cfg,
        weights,
        images,
        batch_size=4,
        out_path='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 检查输出目录是否存在，若存在则将其全部删除，然后新建一个
    if os.path.exists(out_path):
        shutil.rmtree(out_path)  # delete output folder
    os.makedirs(out_path)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Load weights
    # 加载权重永远都是两种方式，要么加载自己之前的训练结果，要么加载官方给出的预训练权重
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Set Dataloader
    base_path = '/home/duanjiaxin/datasets/VOCdevkit/VOC2012'
    # data_path = join(base_path, parse_data_cfg(data_cfg)['valid'])  # Data path
    data_path = parse_data_cfg(data_cfg)['valid']  # Data path
    dataset = LoadImagesAndLabels(data_path, img_size=img_size, augment=True, base_path=base_path)  # Dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=dataset.collate_fn)  # Dataloader

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])  # ["class1", "class2", ..., "classn"]
    # print(classes)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]  # 给每个类一个随机的颜色，类和颜色之间是按次序对应的
    colors = [
        (255, 0, 0), (255, 125, 0), (255, 255, 0), (255, 0, 125), (255, 0, 250),
        (255, 125, 125), (255, 125, 250), (125, 125, 0), (0, 255, 125), (255, 0, 0),
        (0, 0, 255), (125, 0, 255), (0, 125, 255), (0, 255, 255), (125, 125, 255),
        (0, 255, 0), (125, 255, 125), (255, 255, 255), (100, 100, 100), (125, 125, 125),
    ]

    model.eval()
    # 选第 n 批作为测试批次
    n = rm.randint(0, len(dataloader) - 1)  # random 函数选择域为闭区间
    print(n)

    for i, (imgs, tags, path, shape) in enumerate(dataloader):
        if i == n:
            t = time.time()

            # Get detections
            inputs = imgs.to(device)
            infer_outs, train_outs = model(inputs)

            # nms 的输出结果是一个 list，其长度为批大小
            # 并且如果我没猜错的话，它的第 i 个元素就是这个批次中第 i 张图片 bbox 的预测结果
            outputs = non_max_suppression(infer_outs, conf_thres=conf_thres, nms_thres=nms_thres)
            # print(outputs)
            # return

            for i in range(imgs.size(0)):
                # 经非极大抑制后，可得到每张图片的所有包围盒 bboxes，是一个七维向量，其形式为 [x1, y1, x2, y2, conf, cls_conf, c]
                bboxes = outputs[i]
                # print(bboxes)

                # print(bboxes.size())
                if bboxes is not None:
                    # Rescale boxes from 416 to true image size
                    # scale_coords(img_size, bboxes[:, :4], tags.shape).round()

                    # Print results to screen
                    for c in bboxes[:, -1].unique():  # 统计图片 i 中的目标一共有多少个类别（有几种）
                        n = (bboxes[:, -1] == c).sum()
                        print('%s %g' % (classes[int(c)], n), end=' ')
                    print()

                    # Draw bounding boxes and labels of detections
                    # img1 和 img2 是同一张图片的两个复制品，网上面添加的包围框不一样
                    img1 = imgs[i].permute(1, 2, 0).numpy().copy()  # 一定要加 copy() 否则 cv2 报错
                    img2 = imgs[i].permute(1, 2, 0).numpy().copy()  # 一定要加 copy() 否则 cv2 报错
                    # boxes = xywh2xyxy(tags[tags[:, 0] == i, 2:6]).T * img_size

                    plt.subplot(121)
                    plot_boxes(img1, tags[tags[:, 0] == i], img_size=img_size, colors=colors, classes=classes)
                    plt.imshow(img1)
                    plt.title('label')
                    plt.axis('off')
                    plt.subplot(122)
                    plot_pred_boxes(img2, bboxes.cpu(), img_size=img_size, colors=colors, classes=classes)
                    plt.imshow(img2)
                    plt.title('predict')
                    plt.axis('off')
                    plt.show()

                    # if save_images:  # Save generated image with detections
                    #     save_path = out_path + os.sep + path[i].split('/')[-1]
                    #     print(path)
                    #     cv2.imwrite(save_path, img)

                # print('Done. (%.3fs)' % (time.time() - t))
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
