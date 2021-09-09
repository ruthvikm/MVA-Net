#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020/10/01
# @Author  : jet li
# @Email   : jet_uestc@hotmail.com
# @File    : MVA_Net.py
# @SoftWare: PyCharm


import math

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import alexnet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class TaylorSoftmax(nn.Module):
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


class MVAChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(MVAChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, front, rear, left, right):
        # A.size()=[1,64,1,1] B.size()=[1,64,1,1] C=torch.cat((A,B), 1) C.size() = [1,128,1,1]
        b, c, w, h = front.size()
        x1 = torch.cat((self.avg_pool(front), self.avg_pool(rear), self.avg_pool(left), self.avg_pool(right)), 1)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x1))))

        x2 = torch.cat((self.max_pool(front), self.max_pool(rear), self.max_pool(left), self.max_pool(right)), 1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x2))))

        out = avg_out + max_out
        out = self.sigmoid(out)
        front_out, rear_out, left_out, right_out = out.split(c, 1)

        return front_out, rear_out, left_out, right_out


class MVASpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(MVASpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv_front = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv_rear  = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv_left  = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv_right = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv_fusion = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        self.taylorSoftmax = TaylorSoftmax(dim=1, n=4)  # (batch, width ,height) every width taylorSoftmax
        self.sigmoid = nn.Sigmoid()

    def forward(self, front, rear, left, right):
        avg_front = torch.mean(front, dim=1, keepdim=True)
        avg_rear  = torch.mean(rear,  dim=1, keepdim=True)
        avg_left  = torch.mean(left,  dim=1, keepdim=True)
        avg_right = torch.mean(right, dim=1, keepdim=True)

        max_front, _ = torch.max(front, dim=1, keepdim=True)
        max_rear, _  = torch.max(rear,  dim=1, keepdim=True)
        max_left, _  = torch.max(left,  dim=1, keepdim=True)
        max_right, _ = torch.max(right, dim=1, keepdim=True)

        x_front = torch.cat([avg_front, max_front], dim=1)
        x_rear  = torch.cat([avg_rear, max_rear], dim=1)
        x_left  = torch.cat([avg_left, max_left], dim=1)
        x_right = torch.cat([avg_right, max_right], dim=1)

        x_front = self.sigmoid(self.conv_front(x_front))
        x_rear  = self.sigmoid(self.conv_front(x_rear))
        x_left  = self.sigmoid(self.conv_front(x_left))
        x_right = self.sigmoid(self.conv_front(x_right))

        b1, c1, h1, w1 = x_front.size()

        x = torch.cat([x_front, x_rear, x_left, x_right], 2)
        x = self.conv_fusion(x)
        b, c, h, w = x.size()
        x = x.reshape(b, h, w)

        x = self.taylorSoftmax(x)
        front_out, rear_out, left_out, right_out = x.split(h1, dim=1) # h = 4*w origin 224*224

        front_out = front_out.reshape(b1, c1, h1, w1)
        rear_out  = rear_out.reshape(b1, c1, h1, w1)
        left_out  = left_out.reshape(b1, c1, h1, w1)
        right_out = right_out.reshape(b1, c1, h1, w1)

        return front_out, rear_out, left_out, right_out


class SVA_Body(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(SVA_Body, self).__init__()
        planes = in_channels * 4
        self.ca = MVAChannelAttention(planes)
        self.sa = MVASpatialAttention()

    def forward(self, front_sva, rear_sva, left_sva, right_sva):
        (ca_front, ca_rear, ca_left, ca_right) = self.ca(front_sva, rear_sva, left_sva, right_sva)

        ca_front_out = ca_front * front_sva
        ca_rear_out  = ca_rear  * rear_sva
        ca_left_out  = ca_left  * left_sva
        ca_right_out = ca_right * right_sva

        (sa_front, sa_rear, sa_left, sa_right) = self.sa(ca_front_out, ca_rear_out, ca_left_out, ca_right_out)

        sa_front_out = sa_front * ca_front_out
        sa_rear_out  = sa_rear  * ca_rear_out
        sa_left_out  = sa_left  * ca_left_out
        sa_right_out = sa_right * ca_right_out

        return sa_front_out, sa_rear_out, sa_left_out, sa_right_out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, front_downsample=None, rear_downsample=None, left_downsample=None,
                 right_downsample=None):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # front
        self.front_conv1 = conv3x3(inplanes, planes, stride)
        self.front_bn1 = nn.BatchNorm2d(planes)
        self.front_conv2 = conv3x3(planes, planes)
        self.front_bn2 = nn.BatchNorm2d(planes)

        # rear
        self.rear_conv1 = conv3x3(inplanes, planes, stride)
        self.rear_bn1 = nn.BatchNorm2d(planes)
        self.rear_conv2 = conv3x3(planes, planes)
        self.rear_bn2 = nn.BatchNorm2d(planes)

        # left
        self.left_conv1 = conv3x3(inplanes, planes, stride)
        self.left_bn1 = nn.BatchNorm2d(planes)
        self.left_conv2 = conv3x3(planes, planes)
        self.left_bn2 = nn.BatchNorm2d(planes)

        # right
        self.right_conv1 = conv3x3(inplanes, planes, stride)
        self.right_bn1 = nn.BatchNorm2d(planes)
        self.right_conv2 = conv3x3(planes, planes)
        self.right_bn2 = nn.BatchNorm2d(planes)

        # sva body
        self.sva_bn = nn.BatchNorm2d(planes)
        self.sva_body = SVA_Body(in_channels=planes, out_channels=planes, norm_layer=self.sva_bn)

        self.front_downsample = front_downsample
        self.rear_downsample  = rear_downsample
        self.left_downsample  = left_downsample
        self.right_downsample = right_downsample

    def forward(self, input):
        front_in, rear_in, left_in, right_in = torch.chunk(input, 4, dim=0)
        front_in = front_in.squeeze(0)
        rear_in  = rear_in.squeeze(0)
        left_in  = left_in.squeeze(0)
        right_in = right_in.squeeze(0)
        # front
        front_residual = front_in
        front_out = self.front_conv1(front_in)
        front_out = self.front_bn1(front_out)
        front_out = self.relu(front_out)
        front_out = self.front_conv2(front_out)
        front_out = self.front_bn2(front_out)

        # rear
        rear_residual = rear_in
        rear_out = self.rear_conv1(rear_in)
        rear_out = self.rear_bn1(rear_out)
        rear_out = self.relu(rear_out)
        rear_out = self.rear_conv2(rear_out)
        rear_out = self.rear_bn2(rear_out)

        # left
        left_residual = left_in
        left_out = self.left_conv1(left_in)
        left_out = self.left_bn1(left_out)
        left_out = self.relu(left_out)
        left_out = self.left_conv2(left_out)
        left_out = self.left_bn2(left_out)

        # right
        right_residual = right_in
        right_out = self.right_conv1(right_in)
        right_out = self.right_bn1(right_out)
        right_out = self.relu(right_out)
        right_out = self.right_conv2(right_out)
        right_out = self.right_bn2(right_out)

        front_out, rear_out, left_out, right_out = self.sva_body(front_out, rear_out, left_out, right_out)

        if self.front_downsample is not None:
            front_residual = self.front_downsample(front_in)
            rear_residual  = self.rear_downsample(rear_in)
            left_residual  = self.left_downsample(left_in)
            right_residual = self.right_downsample(right_in)

        # total output
        front_out += front_residual
        front_out = self.relu(front_out)

        rear_out  += rear_residual
        rear_out  = self.relu(rear_out)

        left_out  += left_residual
        left_out  = self.relu(left_out)

        right_out += right_residual
        right_out = self.relu(right_out)

        out = torch.stack((front_out, rear_out, left_out, right_out), dim=0)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.front_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.rear_conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        self.left_conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        self.right_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)

        self.front_bn1 = nn.BatchNorm2d(64)
        self.rear_bn1  = nn.BatchNorm2d(64)
        self.left_bn1  = nn.BatchNorm2d(64)
        self.right_bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(block, 64, layers[0])
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2)

        # conv2
        self.front_conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rear_conv2  = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.left_conv2  = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.right_conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.front_fc = nn.Linear(512 * block.expansion, num_classes)
        self.rear_fc  = nn.Linear(512 * block.expansion, num_classes)
        self.left_fc  = nn.Linear(512 * block.expansion, num_classes)
        self.right_fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1):
        front_downsample = None
        rear_downsample  = None
        left_downsample  = None
        right_downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            front_downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            rear_downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            left_downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            right_downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, front_downsample, rear_downsample, left_downsample, right_downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, front_in, rear_in, left_in, right_in):
        # input layer 7*7
        front_out = self.front_conv1(front_in)
        rear_out  = self.rear_conv1(rear_in)
        left_out  = self.left_conv1(left_in)
        right_out = self.right_conv1(right_in)

        # bn1
        front_out = self.front_bn1(front_out)
        rear_out  = self.rear_bn1(rear_out)
        left_out  = self.left_bn1(left_out)
        right_out = self.right_bn1(right_out)
        # relu
        front_out = self.relu(front_out)
        rear_out  = self.relu(rear_out)
        left_out  = self.relu(left_out)
        right_out = self.relu(right_out)
        # maxpool
        front_out = self.maxpool(front_out)
        rear_out  = self.maxpool(rear_out)
        left_out  = self.maxpool(left_out)
        right_out = self.maxpool(right_out)

        # block
        tmp = torch.stack((front_out, rear_out, left_out, right_out), dim=0)
        tmp = self.layer1(tmp)
        tmp = self.layer2(tmp)
        tmp = self.layer3(tmp)
        tmp = self.layer4(tmp)
        front_out, rear_out, left_out, right_out = torch.chunk(tmp, 4, dim=0)
        front_out = front_out.squeeze(0)
        rear_out  = rear_out.squeeze(0)
        left_out  = left_out.squeeze(0)
        right_out = right_out.squeeze(0)

        front_out = self.front_conv2(front_out)
        rear_out  = self.rear_conv2(rear_out)
        left_out  = self.left_conv2(left_out)
        right_out = self.right_conv2(right_out)

        front_out = self.avgpool(front_out)
        rear_out  = self.avgpool(rear_out)
        left_out  = self.avgpool(left_out)
        right_out = self.avgpool(right_out)

        front_out = front_out.view(front_out.size(0), -1)
        rear_out  = rear_out.view(rear_out.size(0), -1)
        left_out  = left_out.view(left_out.size(0), -1)
        right_out = right_out.view(right_out.size(0), -1)

        # output
        front_out = self.front_fc(front_out)
        rear_out  = self.rear_fc(rear_out)
        left_out  = self.left_fc(left_out)
        right_out = self.right_fc(right_out)

        return front_out, rear_out, left_out, right_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        count1 = 0
        count2 = 0
        for k, v in pretrain_dict.items():
            # category 1: keyvalue in the front of string
            key_value1 = ["front_", "rear_", "left_", "right_"]
            for value1 in key_value1:
                new_k = value1 + k
                if new_k in state_dict:
                    model_dict[new_k] = v
                    count1 += 1

            # category 2: keyvalue in the middle of string
            for value2 in key_value1:
                new_k = k[0:9] + value2 + k[9:]
                if new_k in state_dict:
                    model_dict[new_k] = v
                    count2 += 1

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def resnet34_mva(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])

    return model


class MVA_Net(nn.Module):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        super(MVA_Net, self).__init__()

        self.front = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.front_lstm = nn.LSTM(1024, 128, 4)

        self.rear = nn.Sequential(
            nn.Linear(1000, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )
        self.rear_lstm = nn.LSTM(64, 64, 4)

        self.left = nn.Sequential(
            nn.Linear(1000, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )
        self.left_lstm = nn.LSTM(64, 64, 4)
        self.right = nn.Sequential(
            nn.Linear(1000, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )
        self.right_lstm = nn.LSTM(64, 64, 4)
        self.map_backbone = alexnet(pretrained=True)
        self.map = nn.Sequential(
            nn.Linear(1000, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.resnet34_mva = resnet34_mva(pretrained=True)
        self.steer = nn.Sequential(
            nn.Linear(1473, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1)
        )

        self.speed = nn.Sequential(
            nn.Linear(1473, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, front_in, rear_in, left_in, right_in, map_in, speed_in):
        # self define
        batch_size = self.batch_size

        front, rear, left, right = self.resnet34_mva(front_in, rear_in, left_in, right_in)

        # speed_in
        speed_feature = speed_in  #self.speed_in(speed_in)
        # map
        tomtom_map = self.map_backbone(map_in)
        map_feature = self.map(tomtom_map)

        # sight
        front_feature = self.front(front)
        front_feature = torch.stack(
            tuple([front_feature[x:x + batch_size * 4:batch_size] for x in range(batch_size)])).permute(1, 0, 2)
        front_feature_lstm, (hn, cn) = self.front_lstm(front_feature)

        rear_feature = self.rear(rear)
        rear_feature = torch.stack(
            tuple([rear_feature[x:x + batch_size * 4:batch_size] for x in range(batch_size)])).permute(1, 0, 2)
        rear_feature, (hn, cn) = self.rear_lstm(rear_feature)

        left_feature = self.left(left)
        left_feature = torch.stack(
            tuple([left_feature[x:x + batch_size * 4:batch_size] for x in range(batch_size)])).permute(1, 0, 2)
        left_feature, (hn, cn) = self.left_lstm(left_feature)

        right_feature = self.right(right)
        right_feature = torch.stack(
            tuple([right_feature[x:x + batch_size * 4:batch_size] for x in range(batch_size)])).permute(1, 0, 2)
        right_feature, (hn, cn) = self.right_lstm(right_feature)

        concat = torch.cat(
            (front_feature_lstm[-1], front_feature[-1], rear_feature[-1], left_feature[-1], right_feature[-1],
             map_feature, speed_feature
             ), 1).squeeze(1)

        steer = self.steer(concat)
        speed = self.speed(concat)

        return steer, speed


if __name__ == '__main__':
    front_in = torch.ones(32, 3, 224, 224)
    rear_in  = 2 * torch.ones(32, 3, 224, 224)
    left_in  = 3 * torch.ones(32, 3, 224, 224)
    right_in = 4 * torch.ones(32, 3, 224, 224)
    map_in   = 5 * torch.ones(8, 3, 224, 224)
    speed_in = torch.ones(8, 1)

    model = MVA_Net(8)
    steer, speed = model(front_in, rear_in, left_in, right_in, map_in, speed_in)

    print("steer size:", steer.size())
    print("speed size:", speed.size())

