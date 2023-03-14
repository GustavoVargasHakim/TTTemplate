import torch.nn as nn
import torch.nn.functional as F


def get_part(model,layer):
    if layer ==1:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1]
    elif layer ==2:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2]
    elif layer ==3:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3]
    elif layer ==4:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3, model.layer4]
    return nn.Sequential(*extractor)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if num_classes == 12:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if num_classes == 12:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) #64
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) #128
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) #256
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) #512
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes) #512

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        if hasattr(self, 'max_pool'):
            out = self.max_pool(out)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = self.layer4(out)
        print(out.shape)
        #out = F.avg_pool2d(out, 4)
        out = self.avg_pool(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        features = out
        out = self.fc(out)
        if feature:
            return out, features
        else:
            return out

def resnet50(num_classes = 10, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return model

'''
x = torch.rand(1,3,32,32)
model = resnet50(num_classes = 10,  projectors=(True, True, True, True))
_, proj = model(x, project=True)'''