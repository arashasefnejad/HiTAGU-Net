import torch, torch.nn as nn, torchvision.models as tv
class ResNet50Truncated(nn.Module):
    def __init__(self, out_channels=1024, pretrained=True):
        super().__init__()
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.reduce = nn.Conv2d(2048, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels); self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.reduce(x); x = self.bn(x); x = self.relu(x); return x
