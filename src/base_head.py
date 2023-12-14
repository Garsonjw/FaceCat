import torch
import torch.nn as nn

import torchvision.models as models

# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_head(nn.Module):
    def __init__(self, dim=1024):
        super(pixel_head, self).__init__()

        self.downsample1 = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        resnet = models.resnet18(pretrained=True)
        self.layers2 = resnet.layer1
        self.layers3 = resnet.layer2
        self.layers4 = resnet.layer3
        self.layers5 = resnet.layer4

        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(in_features=512, out_features=1)

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        x = self.downsample1(x)
        x = self.downsample2(x)

        feats_x = self.layers2(x)#32*32
        x = self.layers3(feats_x)#16*16
        x = self.layers4(x)# 8*8
        x = self.layers5(x)# 4*4
        x = self.avgpool_2(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = nn.Sigmoid()(x)
        return x
