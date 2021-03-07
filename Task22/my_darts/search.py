import nni
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from nni.algorithms.nas.pytorch.darts import DartsTrainer

from cells import DartsStackedCells
from datasets import getDatasets
from utils import getAccuracy
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def getArguments():
    parser = argparse.ArgumentParser('DARTS-Search')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--channels', default=16, type=int)
    parser.add_argument('--layers', default=8, type=int)
    parser.add_argument('--unrolled', default=False, action='store_true')
    parser.add_argument('--visualization', default=False, action='store_true')
    args = parser.parse_args()
    return args


args = getArguments()
model = DartsStackedCells(3, args.channels, 10, args.layers)
model = model.to(device)
trainset, testset = getDatasets()


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.001)

    trainer = DartsTrainer(
        model,
        criterion,
        lambda outputs, labels: getAccuracy(outputs, labels),
        optimizer,
        args.epochs,
        trainset,
        testset,
        batch_size=args.batch_size,
        unrolled=args.unrolled,
        callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint('./checkpoints')]
    )
    if args.visualization:
        trainer.enable_visualization()
    trainer.train()
