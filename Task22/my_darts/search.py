import logging
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import *

logger = logging.getLogger('darts-nni')
parser = argparse.ArgumentParser('darts-nni')


def init_parser():
    global parser
    parser.add_argument('--layers', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--log-frequency', default=10, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--channels', default=16, type=int)
    parser.add_argument('--unrolled', default=False, action='store_true')
    parser.add_argument('--visualization', default=False, action='store_true')
    parser.add_argument('--v1', default=False, action='store_true')
    parser.add_argument('--v2', default=False, action='store_true')
    return parser.parse_args()


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        height, width = img.size(1), img.size(2)  # done after RandomHorizontalFlip
        mask = np.ones((height, width))  # maybe wrong here
        # mask = np.ones((height, width), np.float32)
        y = np.random.randint(height)
        x = np.random.randint(width)
        x1 = np.clip(x - self.length // 2, 0, width)
        x2 = np.clip(x + self.length // 2, 0, width)
        y1 = np.clip(y - self.length // 2, 0, height)
        y2 = np.clip(y + self.length // 2, 0, height)
        mask[y1: y2, x1: x2] = 0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_cifar10_dataset(cutout_length=0):
    random_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    cutout = [Cutout(cutout_length)] if cutout_length > 0 else []
    train_transform = transforms.Compose(random_transform + normalize_transform + cutout)
    test_transform = transforms.Compose(normalize_transform)
    train_set = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    return train_set, test_set

def get_accuracy(outputs, labels, topk=(1, )):
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = outputs.topk(maxk, 1)
    pred = pred.t()
    if labels.ndimension() > 1:
        labels = labels.max(1)[1]
    correct = pred.eq(labels.view(1, -1)).expand_as(pred)

    res = {}
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res['acc{}'.format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

def main():
    args = init_parser()
    train_set, test_set = get_cifar10_dataset()
    model = CNN(32, 3, args['channels'], 10, args['layers'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], 0.001)

    if args['v1']:
        from nni.algorithms.nas.pytorch.darts import DartsTrainer
        import nni.nas.pytorch.callbacks as callbacks
        trainer = DartsTrainer(
            workers=2,
            model=model,
            loss=criterion,
            metrics=lambda outputs, labels: get_accuracy(outputs, labels),
            optimizer=optim,
            num_epochs=args['epochs'],
            dataset_train=train_set,
            dataset_valid=test_set,
            batch_size=args['batch_size'],
            log_frequency=args['log_frequency'],
            unrolled=args['unrolled'],
            callbacks=[callbacks.LRSchedulerCallback(lr_scheduler), callbacks.ArchitectureCheckpoint('./checkpoints')]
        )
        if args['visualization']:
            trainer.enable_visualization()
        trainer.train()
    elif args['v2']:
        from nni.retiarii.trainer.pytorch import DartsTrainer
        trainer = DartsTrainer(
            workers=2,
            model=model,
            loss=criterion,
            metrics=lambda outputs, labels: get_accuracy(outputs, labels),
            optimizer=optim,
            num_epochs=args['epochs'],
            dataset=train_set,
            batch_size=args['batch_size'],
            log_frequency=args['log_frequency'],
            unrolled=args['unrolled']
        )
        trainer.fit()
        final_architecture = trainer.export()
        print('Final architecture: ', final_architecture)
        with open('checkpoint.json', 'w') as f:
            json.dump(final_architecture, f)


if __name__ == '__main__':
    main()
