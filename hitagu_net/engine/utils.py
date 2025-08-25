import os, yaml, torch, random, numpy as np
from types import SimpleNamespace
def load_config(path):
    with open(path,'r') as f: cfg = yaml.safe_load(f)
    if 'inherit' in cfg:
        base = load_config(cfg['inherit']); base.update(cfg); cfg = base
    return SimpleNamespace(**cfg)
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk); _, pred = output.topk(maxk, 1, True, True); pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)); res=[]
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / target.size(0)).item())
        return res
