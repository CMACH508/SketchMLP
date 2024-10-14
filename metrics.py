import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""  # [128, 10],128
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # [128, 5],indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 5,128

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def class_accuracy(output, target, topk=1):
    """Computes the precision@k for the specified values of k"""  # [128, 10],128
    maxk = topk
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # [128, 1],indices
    pred = pred.t().squeeze(0)
    # print(pred)
    # print(target)
    res = []
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in range(2):
        indices=torch.where(target==k)
        # print(indices)
        correct = torch.where(pred[indices]==k)
        # print(len(correct[0]),len(indices[0]))
        try:
            res.append(len(correct[0])*100.0 / len(indices[0]))
        except:
            res.append(0)
    return res
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
