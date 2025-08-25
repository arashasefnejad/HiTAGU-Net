import torch, torch.nn as nn
class ClassifierHead(nn.Module):
    def __init__(self, in_channels:int, num_classes:int, drop:float=0.5):
        super().__init__(); self.pool = nn.AdaptiveAvgPool2d(1); self.dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_channels, in_channels//2); self.relu = nn.ReLU(inplace=True); self.fc2 = nn.Linear(in_channels//2, num_classes)
    def forward(self, x, return_feats=False):
        x = self.pool(x).flatten(1); f = self.dropout(self.relu(self.fc1(x))); logits = self.fc2(f)
        return (logits, f) if return_feats else logits
