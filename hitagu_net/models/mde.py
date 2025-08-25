import torch, torch.nn as nn
class MotionDifferenceEmbedding(nn.Module):
    def __init__(self, channels:int):
        super().__init__(); self.conv = nn.Conv2d(channels, channels, 1); self.sigmoid = nn.Sigmoid()
    def forward(self, F_t, F_tm1):
        if F_tm1 is None: return F_t, torch.zeros_like(F_t)
        delta = F_t - F_tm1; gate = self.sigmoid(self.conv(delta)); F_hat = F_t + gate * delta; return F_hat, gate
