import argparse
import time

from os.path import join

import torch
from torch.utils.data import DataLoader

from valid import valid
from model import *
from utils.datasets import *
from utils.utils import *

"""
训练
"""


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        num_workers=4,
        transfer=False  # Transfer learning (train only YOLO layers)
):
    # 训练期间参数的保存地址
    weights = 'weights' + os.sep  # os.sep 就是 / 或者 \
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义模型
    model = Darknet(cfg, img_size).to(device)
    # model_info(model)  # 打印模型信息

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 预训练相关参数
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

    # 下面这段都是和预训练相关的
    if resume:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(weights + 'yolov3.pt', map_location=device)
            model.load_state_dict({
                k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255
            }, strict=False)
            # 如果是迁移学习，则前面的参数都不变，只训练 yolo 层的参数
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False  # 找到 yolo 层，只对该层求梯度以更新参数，其它层不要梯度、不更新参数

        else:  # resume from latest.pt
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1  # 比如说上次训练了 50 轮，这次就从第 51 轮开始训练
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        del chkpt

    # 我们用的是这个，官方给出的 darknet 的训练好的权重
    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = model.load_darknet_weights(weights + 'yolov3-tiny.conv.15')
            # cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = model.load_darknet_weights(weights + 'darknet53.conv.74')
            # cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # Data path
    base_path = '/home/duanjiaxin/datasets/VOCdevkit/VOC2012'
    # train_path = join(base_path, parse_data_cfg(data_cfg)['train'])
    # valid_path = join(base_path, parse_data_cfg(data_cfg)['valid'])
    train_path = parse_data_cfg(data_cfg)['train']
    valid_path = parse_data_cfg(data_cfg)['valid']

    # Dataset
    trainset = LoadImagesAndLabels(train_path, img_size=img_size, augment=True, base_path=base_path)
    validset = LoadImagesAndLabels(valid_path, img_size=img_size, augment=True, base_path=base_path)

    # Dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=trainset.collate_fn)
    validloader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=trainset.collate_fn)
    # print(len(trainloader), len(validloader))

    # 记录器
    losses, results = [], []

    # Start training
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
        # 必须每一轮训练的时候都要声明一下，否则在验证时就被置为 eval() 了！
        model.train()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        # 还是跟预训练相关，根据所使用的 yolo 的类型，第 1 轮中不训练前 cutoff 层，从第一轮开始才训练模型的所有层
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                print(name)
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # 正式训练
        mloss = defaultdict(float)  # mean loss

        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
        for i, (imgs, targets, _, _) in enumerate(trainloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            nt = len(targets)
            if nt == 0:  # if no targets continue
                continue

            # Run model
            pred_list = model(imgs)  # 一个列表：[[1, 3, 13, 13, 85], [1, 3, 26, 26, 85], [1, 3, 52, 52, 85]]
            # Build targets
            target_list = build_targets(model, targets)

            # Compute loss
            loss, loss_dict = compute_loss(pred_list, target_list, device=device)

            # Compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * i + val) / (i + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch + 1, epochs), '%g/%g' % (i + 1, len(trainloader)),
                mloss['xy'], mloss['wh'], mloss['conf'], mloss['cls'],
                mloss['total'], nt, time.time() - t_start)
            print(s)

        # Calculate mAP
        # 验证
        with torch.no_grad():
            mP, mR, mAP, mF1, val_loss = valid(model=model, dataloader=validloader, img_size=img_size)
            print(('%10s' * 5) % ('mP', 'mR', 'mAP', 'mF1', 'val loss'))
            print(('%10.4f' * 5) % (mP, mR, mAP, mF1, val_loss))
        losses.append((mloss['total'], val_loss))
        results.append((mP, mR, mAP, mF1))
        # # return
        #
        # 保存模型
        # Update best loss
        if val_loss < best_loss:
            best_loss = val_loss

        # Save training results
        save = True and not args.nosave
        if save:
            # Create checkpoint
            chkpt = {
                'epoch': epoch,
                'best_loss': best_loss,
                'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == val_loss:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if (epoch + 1) % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % (epoch + 1))
                print('Saved the chkpt..')

            # Delete checkpoint
            del chkpt

    return losses, results


"""
训练参数
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')  # 10
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')  # 16
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    args = parser.parse_args()
    print(args, end='\n\n')

    init_seeds()

    loss, res = train(
        args.cfg,
        args.data_cfg,
        img_size=args.img_size,
        resume=args.resume or args.transfer,
        transfer=args.transfer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate=args.accumulate,
        multi_scale=args.multi_scale,
        num_workers=args.num_workers
    )

    plt.figure()
    plt.subplot(121)
    plt.plot(range(len(loss)), [l[0] for l in loss], label='tr_loss')
    plt.plot(range(len(loss)), [l[1] for l in loss], label='val_loss')
    plt.legend()
    plt.title('Loss curve of YOLOv3')
    plt.xlabel('Iteras')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.plot(range(len(res)), [r[0] for r in res], label='mp')
    plt.plot(range(len(res)), [r[1] for r in res], label='mr')
    plt.plot(range(len(res)), [r[2] for r in res], label='map')
    plt.plot(range(len(res)), [r[3] for r in res], label='mf1')
    plt.legend()
    plt.title('Valid of YOLOv3')
    plt.xlabel('Iteras')
    plt.ylabel('Value')
    plt.show()
