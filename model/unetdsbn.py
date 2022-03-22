# -*- coding: utf-8 -*-
"""
2D Unet-like architecture code in Pytorch
"""
import math
import numpy as np
from model.layers import *
from model.dsbn import DomainSpecificBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch

class MyUpsample2(nn.Module):
    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)


def normalization(planes, norm='gn', num_domains=None, momentum=0.1):
    if norm == 'dsbn':
        m = DomainSpecificBatchNorm2d(planes, num_domains=num_domains, momentum=momentum)
    elif norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m

#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, num_domains=None, momentum=0.1):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm, num_domains, momentum=momentum)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm, num_domains, momentum=momentum)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm, num_domains, momentum=momentum)

    def forward(self, x, weights=None, layer_idx=None, domain_label=None):

        if weights == None:
            weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
            weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
            weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']

        if not self.first:
            x = maxpool2D(x, kernel_size=2)

        #layer 1 conv, bn
        x = conv2d(x, weight_1, bias_1)
        if domain_label is not None:
            x, _ = self.bn1(x, domain_label)
        else:
            x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = conv2d(x, weight_2, bias_2)
        if domain_label is not None:
            y, _ = self.bn2(y, domain_label)
        else:
            y = self.bn2(y)
        y = relu(y)

        #layer 3 conv, bn
        z = conv2d(y, weight_3, bias_3)
        if domain_label is not None:
            z, _ = self.bn3(z, domain_label)
        else:
            z = self.bn3(z)
        z = relu(z)

        return z

class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, num_domains=None, momentum=0.1):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm, num_domains, momentum=momentum)

        self.pool = MyUpsample2()
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm, num_domains, momentum=momentum)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm, num_domains, momentum=momentum)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev, weights=None, layer_idx=None, domain_label=None):

        if weights == None:
            if not self.first:
                weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            if not self.first:
                weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
            weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
            weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']
            
        #layer 1 conv, bn, relu
        if not self.first:
            x = conv2d(x, weight_1, bias_1, )
            if domain_label is not None:
                x, _ = self.bn1(x, domain_label)
            else:
                x = self.bn1(x)
            x = relu(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = conv2d(y, weight_2, bias_2, kernel_size=1, stride=1, padding=0)
        if domain_label is not None:
            y, _ = self.bn2(y, domain_label)
        else:
            y = self.bn2(y)
        y = relu(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = conv2d(y, weight_3, bias_3)
        if domain_label is not None:
            y, _ = self.bn3(y, domain_label)
        else:
            y = self.bn3(y)
        y = relu(y)

        return y


class Unet2D(nn.Module):
    def __init__(self, c=1, n=16, norm='bn', num_classes=2, num_domains=4, momentum=0.1):
        super(Unet2D, self).__init__()

        self.convd1 = ConvD(c,     n, norm, first=True, num_domains=num_domains, momentum=momentum)
        self.convd2 = ConvD(n,   2*n, norm, num_domains=num_domains, momentum=momentum)
        self.convd3 = ConvD(2*n, 4*n, norm, num_domains=num_domains, momentum=momentum)
        self.convd4 = ConvD(4*n, 8*n, norm, num_domains=num_domains, momentum=momentum)
        self.convd5 = ConvD(8*n,16*n, norm, num_domains=num_domains, momentum=momentum)

        self.convu4 = ConvU(16*n, norm, first=True, num_domains=num_domains, momentum=momentum)
        self.convu3 = ConvU(8*n, norm, num_domains=num_domains, momentum=momentum)
        self.convu2 = ConvU(4*n, norm, num_domains=num_domains, momentum=momentum)
        self.convu1 = ConvU(2*n, norm, num_domains=num_domains, momentum=momentum)

        self.seg1 = nn.Conv2d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, weights=None, domain_label=None):
        if weights == None:
            x1 = self.convd1(x, domain_label=domain_label)
            x2 = self.convd2(x1, domain_label=domain_label)
            x3 = self.convd3(x2, domain_label=domain_label)
            x4 = self.convd4(x3, domain_label=domain_label)
            x5 = self.convd5(x4, domain_label=domain_label)

            y4 = self.convu4(x5, x4, domain_label=domain_label)
            y3 = self.convu3(y4, x3, domain_label=domain_label)
            y2 = self.convu2(y3, x2, domain_label=domain_label)
            y1 = self.convu1(y2, x1, domain_label=domain_label)

            y1_pred = conv2d(y1, self.seg1.weight, self.seg1.bias, kernel_size=None, stride=1, padding=0)
        else:
            x1 = self.convd1(x, weights=weights, layer_idx='module.convd1', domain_label=domain_label)
            x2 = self.convd2(x1, weights=weights, layer_idx='module.convd2', domain_label=domain_label)
            x3 = self.convd3(x2, weights=weights, layer_idx='module.convd3', domain_label=domain_label)
            x4 = self.convd4(x3, weights=weights, layer_idx='module.convd4', domain_label=domain_label)
            x5 = self.convd5(x4, weights=weights, layer_idx='module.convd5', domain_label=domain_label)

            y4 = self.convu4(x5, x4, weights=weights, layer_idx='module.convu4', domain_label=domain_label)
            y3 = self.convu3(y4, x3, weights=weights, layer_idx='module.convu3', domain_label=domain_label)
            y2 = self.convu2(y3, x2, weights=weights, layer_idx='module.convu2', domain_label=domain_label)
            y1 = self.convu1(y2, x1, weights=weights, layer_idx='module.convu1', domain_label=domain_label)

            y1_pred = conv2d(y1, weights['module.seg1.weight'], weights['module.seg1.bias'], kernel_size=None, stride=1, padding=0)
        predictions = torch.sigmoid(input=y1_pred)

        return predictions