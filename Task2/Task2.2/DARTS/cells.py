import torch
import torch.nn as nn
import nni
from nni.nas.pytorch.search_space_zoo import DartsCell
from nni.nas.pytorch.search_space_zoo.darts_ops import DropPath


class DartsStackedCells(nn.Module):
    def drop_path_probability(self, p):
        for module in self.modules():
            if isinstance(module, DropPath):
                module.p = p

    def __init__(self, in_channels, channels, n_classes, n_layers, n_nodes=4, stem_multiplier=2):
        super(DartsStackedCells, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        cur_channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, cur_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cur_channels)
        )
        pp_channels, p_channels, cur_channels = cur_channels, cur_channels, channels
        self.cells = nn.ModuleList()
        p_reduction, cur_reduction = False, False
        for i in range(n_layers):
            p_reduction, cur_reduction = cur_reduction, False
            if i in [n_layers // 3, 2 * n_layers // 3]:
                cur_channels *= 2
                cur_reduction = True
            self.cells.append(DartsCell(n_nodes, pp_channels, p_channels, cur_channels, p_reduction, cur_reduction))
            cur_channels_out = cur_channels * n_nodes
            pp_channels, p_channels = p_channels, cur_channels_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(p_channels, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        output = self.gap(s1)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output
