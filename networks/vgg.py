import torchvision
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchvision.models import VGG16_Weights


class VGG16_fc_mapping(nn.Module):
    def __init__(self, pretrain=True):
        super(VGG16_fc_mapping, self).__init__()
        # resnet50
        if pretrain:
            net = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            net = models.vgg16()
        self.net = net
        # self.fc 
        self.net.fc = nn.Linear(512, 7)
        # embed
        self.mapping = nn.Linear(512, 128, bias=False)

    def forward(self, x, need_feature=False, mapping=False):
        x = self.net.features(x)
        # print(x.shape)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.net.fc(x)
        if need_feature:
            if mapping:
                feature = self.mapping(feature)
            return x, feature
        else:
            return x