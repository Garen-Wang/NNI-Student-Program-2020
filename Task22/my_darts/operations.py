import torch
import torch.nn as nn


class DropPath(nn.Module):
    # p: possibility of dropping the path
    def __init__(self, p=0.):
        super(DropPath, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            # only this way to drop path
            mask = torch.zeros((x.size(0), 1, 1, 1), device=x.device).bernoulli_(1 - self.p)
            return x / (1 - self.p) * mask
        else:
            return x


class PoolBN(nn.Module):
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super(PoolBN, self).__init__()
        assert pool_type in ['max', 'avg']
        self.pool = nn.MaxPool2d(kernel_size, stride, padding) if pool_type == 'max' else nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        output = self.pool(x)
        output = self.bn(x)
        return output


class StandardConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(StandardConv, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(FactorizedConv, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, 1, kernel_size, stride, padding, bias=False),
            nn.Conv2d(1, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DillatedConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DillatedConv, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SeparableConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SeparableConv, self).__init__()
        self.net = nn.Sequential(
            DillatedConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DillatedConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, 2, 0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, 2, 0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        x = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        x = self.bn(x)
        return x

