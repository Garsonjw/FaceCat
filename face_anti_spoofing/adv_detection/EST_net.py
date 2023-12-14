import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class Conv2d_basic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_basic, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        return out_normal


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # change your path
    model_path = '/data/home/scv7305/run/chenjiawei/adv_branch/CDCN/models/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    # print(model)
    return model


class Resnet18(nn.Module):

    def __init__(self, basic_conv=Conv2d_basic, map_size=16, input_channel=3):
        super(Resnet18, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        # Original

        self.downconv1 = nn.Sequential(
            basic_conv(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            basic_conv(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.downconv2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.lastconv1 = nn.Sequential(
            basic_conv(448, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.average = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spoofer = nn.Linear(map_size * map_size, 1)

    def forward(self, x):  # x [3, 112, 112]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_Block1 = self.layer1(x)
        x_Block1_32x32 = self.downconv1(x_Block1)

        x_Block2 = self.layer2(x_Block1)
        x_Block2_32x32 = self.downconv2(x_Block2)

        x_Block3 = self.layer3(x_Block2)

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3), dim=1)

        embedding = self.lastconv1(x_concat)
        embedding_1 = self.average(embedding)
        embedding_2 = self.average(embedding_1)
        map_x_2 = self.decoder3(embedding_2)
        map_x_2 = map_x_2.squeeze(1)

        return map_x_2