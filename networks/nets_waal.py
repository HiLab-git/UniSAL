import torchvision
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable, grad


class ResNet50_waal(nn.Module):
    def __init__(self, pretrain=True, num_classes=7):
        super(ResNet50_waal, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, num_classes)
    
        self.dis = Discriminator(2048)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.net.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc(x)
        return x, feature
    
    def gradient_penalty(self, h_s, h_t):
        ''' Gradeitnt penalty approach'''
        alpha = torch.rand(h_s.size(0), 1).cuda()
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
        # interpolates.requires_grad_()
        preds = self.dis(interpolates)
        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()

        return gradient_penalty


class Discriminator(nn.Module):
        """Adversary architecture(Discriminator) for WAE-GAN."""
        def __init__(self, dim=32):
                super(Discriminator, self).__init__()
                self.dim = np.prod(dim)
                self.net = nn.Sequential(
                        nn.Linear(self.dim, 512),
                        nn.ReLU(True),
                        nn.Linear(512, 512),
                        nn.ReLU(True),
                        nn.Linear(512, 1),
                        nn.Sigmoid(),
                )
                self.weight_init()

        def weight_init(self):
                for block in self._modules:
                        for m in self._modules[block]:
                                kaiming_init(m)

        def forward(self, z):
                return self.net(z).reshape(-1)

def kaiming_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                        m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                        m.bias.data.fill_(0)

def normal_init(m, mean, std):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.data.normal_(mean, std)
                if m.bias.data is not None:
                        m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                if m.bias.data is not None:
                        m.bias.data.zero_()