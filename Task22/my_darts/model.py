import numpy as np
import torch
import torch.nn as nn
from nni.nas.pytorch import mutables
from collections import OrderedDict
import operations

class AuxiliaryHead(nn.Module):
    def __init__(self, input_size, C, n_classes):
        assert input_size in [7, 8]
        super(AuxiliaryHead, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),  # 1x1 conv
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        output = self.net(x)
        output = output.view(output.size(0), -1)
        logits = self.linear(output)
        return logits
    

class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, C, num_downsample_connect):
        super(Node, self).__init__()
        self.operations = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append('{}_p{}'.format(node_id, i))
            self.operations.append(mutables.LayerChoice(OrderedDict([
                ('maxpool3x3', operations.PoolBN('max', C, 3, stride, 1, affine=False)),
                ('avgpool3x3', operations.PoolBN('avg', C, 3, stride, 1, affine=False)),
                ('skipconnect', nn.Identity() if stride == 1 else operations.FactorizedReduce(C, C, affine=False)),
                ('separableconv3x3', operations.SeparableConv(C, C, 3, stride, 1, affine=False)),
                ('separableconv5x5', operations.SeparableConv(C, C, 5, stride, 2, affine=False)),
                ('dillatedconv3x3', operations.DillatedConv(C, C, 3, stride, 1, affine=False)),
                ('dillatedconv5x5', operations.DillatedConv(C, C, 5, stride, 2, affine=False))
            ]), key=choice_keys[-1]))
        # self.drop_path = operations.DropPath()
        self.input_switch = mutables.InputChoice(choose_from=choice_keys, n_chosen=2, key='{}_switch'.format(node_id))

    def forward(self, prev_nodes):
        assert len(self.operations) == len(prev_nodes)
        output = [operation(prev_node) for operation, prev_node in zip(self.operations, prev_nodes)]
        # output = [self.drop_path(x) if x is not None else None for x in output]
        return self.input_switch(output)


class Cell(nn.Module):
    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        if reduction_p:
            self.preproc0 = operations.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = operations.StandardConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = operations.StandardConv(channels_p, channels, 1, 1, 0, affine=False)

        self.mutable_operations = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_operations.append(Node('{}_n{}'.format('reduce' if reduction else 'normal', depth), depth, channels, 2 if reduction else 0))

    def forward(self, s0, s1):
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_operations:
            current_tensor = node(tensors)
            tensors.append(current_tensor)
        output = torch.cat(tensors[2:], dim=1)
        return output


class CNN(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, n_classes, n_layers, n_nodes=4, stem_multiplier=3, auxiliary=False):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        c_cur = stem_multiplier * self.out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur)
        )
        channels_pp, channels_p, c_cur = c_cur, c_cur, out_channels
        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            for i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

            if i == self.aux_pos:
                self.aux_head = AuxiliaryHead(input_size // 4, channels_p, n_classes)

            self.gap = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(channels_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        output = self.gap(s1)
        output = output.view(output.size(0), -1)
        logits = self.linear(output)
        if aux_logits is not None:
            return logits, aux_logits
        return logits

    def set_drop_path_probability(self, p):
        for module in self.modules():
            if isinstance(module, operations.DropPath):
                module.p = p
