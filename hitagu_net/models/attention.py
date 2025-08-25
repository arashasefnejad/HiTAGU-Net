import torch, torch.nn as nn
class TemporalSelfAttention(nn.Module):
    def __init__(self, channels:int, attn_dim:int):
        super().__init__(); self.gap = nn.AdaptiveAvgPool2d(1)
        self.q = nn.Linear(channels, attn_dim); self.k = nn.Linear(channels, attn_dim); self.v = nn.Linear(channels, attn_dim)
        self.proj = nn.Linear(attn_dim, channels)
    def forward(self, feats):
        B,C,H,W = feats[0].shape; tokens = [self.gap(f).flatten(1) for f in feats]
        X = torch.stack(tokens, dim=1); Q = self.q(X); K = self.k(X); V = self.v(X)
        attn = torch.softmax(Q @ K.transpose(1,2) / (K.shape[-1] ** 0.5), dim=-1); Z = self.proj(attn @ V)
        out=[]; 
        for t,f in enumerate(feats):
            zt = Z[:,t,:].view(B,-1,1,1).expand(-1,f.shape[1],H,W); out.append(f + zt)
        return out
class SpatialChannelAttention(nn.Module):
    def __init__(self, channels:int, r:int=16):
        super().__init__(); self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//r); self.relu = nn.ReLU(inplace=True); self.fc2 = nn.Linear(channels//r, channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B,C,H,W = x.shape; s = self.gap(x).flatten(1); s = self.fc1(s); s=self.relu(s); s=self.fc2(s); s=self.sigmoid(s).view(B,C,1,1); return x*s
