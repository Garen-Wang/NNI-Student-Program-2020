import torch


def getAccuracy(outputs, labels, topk=(1,)):
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = outputs.topk(maxk, 1)
    pred = pred.t()
    if labels.ndimension() > 1:
        labels = labels.max(1)[1]
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = {}
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res['acc{}'.format(k)] = correct_k.mul_(1. / batch_size).item()
    return res
