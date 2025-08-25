import torch, torch.nn as nn
from .backbone import ResNet50Truncated
from .mde import MotionDifferenceEmbedding
from .three_state_unit import ThreeStateTemporalUnit
from .attention import TemporalSelfAttention, SpatialChannelAttention
from .classifier import ClassifierHead
class HiTAGUNet(nn.Module):
    def __init__(self, num_classes=700, feature_dim=1024, hidden_dim=512, attn_dim=512, drop=0.5, pretrained_backbone=True):
        super().__init__(); self.backbone = ResNet50Truncated(out_channels=feature_dim, pretrained=pretrained_backbone)
        self.mde = MotionDifferenceEmbedding(channels=feature_dim); self.tu  = ThreeStateTemporalUnit(in_channels=feature_dim, hidden_dim=hidden_dim)
        self.tattn = TemporalSelfAttention(channels=hidden_dim, attn_dim=attn_dim); self.sattn = SpatialChannelAttention(channels=hidden_dim, r=16)
        self.classifier = ClassifierHead(in_channels=hidden_dim, num_classes=num_classes, drop=drop)
    def forward(self, x):
        B,T,C,H,W = x.shape; x = x.view(B*T, C, H, W); feats = self.backbone(x); feats = feats.view(B, T, feats.size(1), feats.size(2), feats.size(3))
        enhanced, gates = [], []
        for t in range(T):
            Ft = feats[:, t]; Ft_1 = feats[:, t-1] if t>0 else None; Fhat, gate = self.mde(Ft, Ft_1); enhanced.append(Fhat); gates.append(gate)
        outputs, H = [], None
        for t in range(T):
            x_t = enhanced[t]; x_tm1 = enhanced[t-1] if t>0 else None; x_tm2 = enhanced[t-2] if t>1 else None; M_t = gates[t]
            y, H = self.tu(x_t, x_tm1, x_tm2, M_t, H); outputs.append(y)
        attn_out = self.tattn(outputs); y_last = self.sattn(attn_out[-1]); logits, feats = self.classifier(y_last, return_feats=True)
        seq_embed = [f.mean(dim=[2,3]) for f in attn_out]; return logits, feats, seq_embed
