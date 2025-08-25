from torch.utils.tensorboard import SummaryWriter
class TBLogger:
    def __init__(self, log_dir): self.w = SummaryWriter(log_dir=log_dir); self.step = 0
    def log_scalars(self, d:dict, step=None, split='train'):
        s = self.step if step is None else step
        for k,v in d.items(): self.w.add_scalar(f"{split}/{k}", v, s)
    def inc(self): self.step += 1
    def close(self): self.w.close()
