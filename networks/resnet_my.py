import torchvision
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class ResNet50_fc(nn.Module):
    def __init__(self, pretrain=True, num_classes=2):
        super(ResNet50_fc, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net
        self.net.fc = nn.Linear(2048, num_classes)

    def forward(self, x, need_feature=False):
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
        if need_feature:
            return x, feature
        else:
            return x


class ResNet50_fc3(nn.Module):
    def __init__(self, pretrain=True):
        super(ResNet50_fc3, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # self.fc 
        self.fc = nn.Linear(1000, 7)

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
        x = self.net.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc(x)
        return x

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = self.net.relu(self.net.bn1(self.net.conv1(x)))
        out_list.append(out)
        out = self.net.layer1(out)
        out_list.append(out)
        out = self.net.layer2(out)
        out_list.append(out)
        out = self.net.layer3(out)
        out_list.append(out)
        out = self.net.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)), padding=0)
        out = out.view(out.size(0), -1)
        out = self.net.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        # print(x.shape)
        y = self.fc(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.net.relu(self.net.bn1(self.net.conv1(x)))
        if layer_index == 1:
            out = self.net.layer1(out)
        elif layer_index == 2:
            out = self.net.layer1(out)
            out = self.net.layer2(out)
        elif layer_index == 3:
            out = self.net.layer1(out)
            out = self.net.layer2(out)
            out = self.net.layer3(out)
        elif layer_index == 4:
            out = self.net.layer1(out)
            out = self.net.layer2(out)
            out = self.net.layer3(out)
            out = self.net.layer4(out)               
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = self.net.relu(self.net.bn1(self.net.conv1(x)))
        out = self.net.layer1(out)
        out = self.net.layer2(out)
        out = self.net.layer3(out)
        penultimate = self.net.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        out = self.net.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        # print(x.shape)
        y = self.fc(out)
        return y, penultimate
    

class Feature_extractor(nn.Module):
    def __init__(self, pretrain=True):
        super(Feature_extractor, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net
        del self.net.fc

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
        x = x.squeeze(3).squeeze(2)
        return x


class ResNet50_multi3(nn.Module):
    def __init__(self, pretrain=True):
        super(ResNet50_multi3, self).__init__()
        
        self.num_classes = 7
        self.classifier_head1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, self.num_classes, 1),
            )
        # self.classifier_head1 = nn.Linear(2048, self.num_classes)
        self.classifier_head2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, self.num_classes, 1),
            )
        self.classifier_head3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, self.num_classes, 1),
            )
        self._initialize_weights()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x1 = self.classifier_head1(x)
        logit1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        logit1 = logit1.view(-1, self.num_classes)

        # x1 = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        # x1 = x1.view(x1.size(0), -1)
        # logit1 = self.classifier_head1(x1)
        # print(logit1)
        
        x2 = self.classifier_head2(x)
        logit2 = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        logit2 = logit2.view(-1, self.num_classes)

        x3 = self.classifier_head3(x)
        logit3 = F.avg_pool2d(x3, kernel_size=(x3.size(2), x3.size(3)), padding=0)
        logit3 = logit3.view(-1, self.num_classes)

        return logit1, logit2, logit3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if "classifier_head" in name:
                if "weight" in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if "weight" in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


class ResNet50_multi3_v2(nn.Module):
    def __init__(self, pretrain=True):
        super(ResNet50_multi3_v2, self).__init__()
        
        self.num_classes = 7
        self.fc1 = nn.Linear(1000, self.num_classes)
        self.fc2 = nn.Linear(1000, self.num_classes)
        self.fc3 = nn.Linear(1000, self.num_classes)

        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        # print(x.shape)
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)

        return logit1, logit2, logit3


class ResNet50_dropout(nn.Module):
    def __init__(self, pretrain=True):
        super(ResNet50_dropout, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.1)
        # self.fc 
        self.fc = nn.Linear(1000, 7)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.dropout1(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        x = self.relu(x)
        x = self.dropout2(x)
        # print(x.shape)
        x = self.fc(x)
        return x


class ResNet50_fc_mapping(nn.Module):
    def __init__(self, pretrain=True, num_classes=7):
        super(ResNet50_fc_mapping, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.net = net
        # self.fc 
        self.net.fc = nn.Linear(2048, num_classes)
        # embed
        self.mapping = nn.Linear(2048, 128, bias=False)


    def forward(self, x, need_feature=False, mapping=False):
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
        if need_feature:
            if mapping:
                feature = self.mapping(feature)
            return x, feature
        else:
            return x


class ResNet18_fc_mapping(nn.Module):
    def __init__(self, pretrain=True):
        super(ResNet18_fc_mapping, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet18()
        self.net = net
        # self.fc 
        self.net.fc = nn.Linear(512, 7)
        # embed
        self.mapping = nn.Linear(512, 128, bias=False)


    def forward(self, x, need_feature=False, mapping=False):
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
        if need_feature:
            if mapping:
                feature = self.mapping(feature)
            return x, feature
        else:
            return x