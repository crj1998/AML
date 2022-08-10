from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        identity = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + identity
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        identity = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = out + identity
        return out


class PreActResNet(nn.Module):
    _C = [3, 64, 128, 256, 512]
    def __init__(self, block, num_blocks, num_classes):
        super(PreActResNet, self).__init__()
        self.channels = self._C[1]
        self.conv = nn.Conv2d(self._C[0], self._C[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_layer(block, self._C[1], num_blocks[0], 1)
        self.layer2 = self.make_layer(block, self._C[2], num_blocks[1], 2)
        self.layer3 = self.make_layer(block, self._C[3], num_blocks[2], 2)
        self.layer4 = self.make_layer(block, self._C[4], num_blocks[3], 2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self._C[4]*block.expansion, num_classes)

    def make_layer(self, block, out_channels, block_num, stride):
        layers = OrderedDict([("block0", block(self.channels, out_channels, stride))])
        self.channels = out_channels * block.expansion
        for i in range(block_num - 1):
            layers[f"block{i+1}"] = block(self.channels, out_channels, 1)
            self.channels = out_channels * block.expansion

        return nn.Sequential(layers)

    def forward(self, x):
        x = self.conv(x)     # (N,  64, 32, 32)
        x = self.layer1(x)     # (N,  64, 32, 32)
        x = self.layer2(x)     # (N, 128, 16, 16)
        x = self.layer3(x)     # (N, 256,  8,  8)
        x = self.layer4(x)     # (N, 512,  4,  4)
        x = F.relu(self.bn(x))
        x = self.avgpool(x)    # (N, 512,  1,  1)
        x = self.flatten(x)    # (N, 512)
        x = self.fc(x)         # (N, 10)
        return x


def preactresnet18(num_classes):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes)

def preactresnet34(num_classes):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes)

def preactresnet50(num_classes):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes)

def preactresnet101(num_classes):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes)

def PreActResNet152(num_classes):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes)


if __name__ == '__main__':
    net = preactresnet50(10)
    x = torch.randn((8, 3, 32, 32))
    y = net(x)
    print(y.shape)