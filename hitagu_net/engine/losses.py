import torch, torch.nn as nn
class CenterLoss(nn.Module):
    def __init__(self, num_classes:int, feat_dim:int, alpha:float=0.5):
        super().__init__(); self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)); self.alpha = alpha; self.feat_dim = feat_dim
    def forward(self, feats, labels):
        batch_centers = self.centers.index_select(0, labels); return ((feats - batch_centers)**2).sum(dim=1).mean()
def temporal_consistency(hidden_seq):
    if len(hidden_seq) < 2: return hidden_seq[0].new_tensor(0.0)
    diffs=[]; 
    for t in range(len(hidden_seq)-1):
        x,y = hidden_seq[t], hidden_seq[t+1]
        if x.dim()==4: x=x.mean(dim=[2,3]); 
        if y.dim()==4: y=y.mean(dim=[2,3])
        diffs.append((x-y).pow(2).mean())
    return torch.stack(diffs).mean()
