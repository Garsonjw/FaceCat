import torchvision
from torch import nn






class DFRAA_fet(nn.Module):

    def __init__(self):
        super(DFRAA_fet, self).__init__()
        model_resnet = torchvision.models.resnet18(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        # Original

    def forward(self, x):  # x [3, 112, 112]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x= self.layer4(x)

        return x