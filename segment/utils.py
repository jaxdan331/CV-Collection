import numpy as np


# 获取混淆矩阵
def fast_hist(a, b, n):
    """
    Return a histogram that's the confusion matrix of a and b
    :param a: np.ndarray with shape (HxW,)
    :param b: np.ndarray with shape (HxW,)
    :param n: num of classes
    :return: np.ndarray with shape (n, n)
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    """
    Calculate the IoU(Intersection over Union) for each class
    :param hist: np.ndarray with shape (n, n)
    :return: np.ndarray with shape (n,)
    """
    np.seterr(divide="ignore", invalid="ignore")
    res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide="warn", invalid="warn")
    res[np.isnan(res)] = 0.
    return res


class ComputeIoU():
    """
    IoU: Intersection over Union
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.cfsmatrix = np.zeros((num_classes, num_classes), dtype="uint64")  # confusion matrix
        self.ious = dict()  # 用字典保存各类别的 IoU 值

    def get_cfsmatrix(self):
        return self.cfsmatrix

    def get_ious(self):
        self.ious = dict(zip(range(self.num_classes), per_class_iou(self.cfsmatrix)))  # {0: iou, 1: iou, ...}
        return self.ious

    def get_miou(self, ignore=None):
        self.get_ious()
        total_iou = 0
        count = 0
        for key, value in self.ious.items():
            if isinstance(ignore, list) and key in ignore or isinstance(ignore, int) and key == ignore:
                continue
            total_iou += value
            count += 1
        return total_iou / count

    # 将一个类变成一个函数（使这个类的实例可以像函数一样调用）
    def __call__(self, pred, label):
        """
        :param pred: [B, H, W]
        :param label:  [B, H, W]
        Channel == 1
        """
        # 这两句是重点
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()

        assert pred.shape == label.shape

        self.cfsmatrix += fast_hist(pred.reshape(-1), label.reshape(-1), self.num_classes).astype("uint64")