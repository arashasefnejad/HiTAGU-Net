import torch, torch.nn as nn
class ThreeStateTemporalUnit(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int):
        super().__init__()
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Conv2d(in_channels*3 + in_channels, hidden_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim); self.act = nn.ReLU(inplace=True)
        self.gi = nn.Conv2d(hidden_dim, hidden_dim, 1); self.gf = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.go = nn.Conv2d(hidden_dim, hidden_dim, 1); self.cc = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.sigmoid = nn.Sigmoid(); self.tanh = nn.Tanh()
    def forward(self, x_t, x_tm1=None, x_tm2=None, M_t=None, H_tm1=None):
        B,C,H,W = x_t.shape; x1 = x_tm1 if x_tm1 is not None else torch.zeros_like(x_t)
        x2 = x_tm2 if x_tm2 is not None else torch.zeros_like(x_t)
        m  = M_t if M_t is not None else torch.zeros_like(x_t); mgp = self.gp(m).expand(-1,-1,H,W)
        z = torch.cat([x_t, x1, x2, mgp], dim=1); z = self.act(self.bn(self.fuse(z)))
        i = self.sigmoid(self.gi(z)); f = self.sigmoid(self.gf(z)); o = self.sigmoid(self.go(z)); c_tilde = self.tanh(self.cc(z))
        if H_tm1 is None: H_tm1 = torch.zeros(B, z.shape[1], H, W, device=z.device, dtype=z.dtype)
        H_t = f*H_tm1 + i*c_tilde; Y_t = o*self.tanh(H_t); return Y_t, H_t
