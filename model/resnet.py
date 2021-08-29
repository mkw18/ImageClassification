# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#ResNet18, 34
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1, groups = 1, base_width = 64):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
# ResNet50, 101
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inchannel, outchannel, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(outchannel * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(width)
        # self.conv2 = conv3x3(width, width, stride, groups)
        # self.bn2 = nn.BatchNorm2d(width)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.stride = stride
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups = groups, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, outchannel * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel * self.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel * self.expansion)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        # identity = x
# 
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
# 
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
# 
        # out = self.conv3(out)
        # out = self.bn3(out)
# 
        # if self.downsample is not None:
        #     identity = self.downsample(x)
# 
        # out += identity
        # out = self.relu(out)
# 
        # return out

class ResNet(nn.Module):
    def __init__(self, Block, layers, groups = 1, base_width = 64, num_classes=20):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.groups = groups
        self.base_width = base_width
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(3, 1, 1)
        )
        self.layer1 = self.make_layer(Block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(Block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(Block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(Block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.Linear(512, 256),
        #     nn.Linear(256, num_classes))
        self.fc = nn.Linear(512 * Block.expansion, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, self.groups, self.base_width))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.avg_pool(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18(num_class = 20):
    return ResNet(ResidualBlock, [2,2,2,2], num_classes = num_class)

def ResNet22(num_class = 20):
    return ResNet(ResidualBlock, [2,3,3,2], num_classes = num_class)

def ResNet34(num_class = 20):
    return ResNet(ResidualBlock, [3,4,6,3], num_classes = num_class)

def ResNet50(num_class = 20):
    return ResNet(Bottleneck, [3,4,6,3], num_classes = num_class)

def ResNet101(num_class = 20):
    return ResNet(Bottleneck, [3,4,23,3], num_classes = num_class)

def ResNet152(num_class = 20):
    return ResNet(Bottleneck, [3,8,36,3], num_classes = num_class)

def ResNext50(num_class = 20):
    return ResNet(Bottleneck, [3,4,6,3], groups=32, base_width=4, num_classes = num_class)

def ResNext101(num_class = 20):
    return ResNet(Bottleneck, [3,4,23,3], groups=32, base_width=8)

def wide_ResNet50(num_class = 20):
    return ResNet(Bottleneck, [3,4,6,3], base_width=64*2)

def wide_ResNet101(num_class = 20):
    return ResNet(Bottleneck, [3,4,23,3], base_width=64*2)