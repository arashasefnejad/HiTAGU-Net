import os, torch, torch.nn as nn, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from ..models import HiTAGUNet
from ..data.kinetics700 import Kinetics700Dataset
from ..data.ssv2 import SSv2Dataset
from .utils import accuracy
from .losses import CenterLoss, temporal_consistency
from .logger import TBLogger


def build_dataset(cfg, train=True):
    if cfg.data['dataset'] in ['K700','Kinetics','Kinetics700']:
        return Kinetics700Dataset(cfg.data['root'],
                                  cfg.data['train_list'] if train else cfg.data['val_list'],
                                  cfg.data['frames_per_clip'],
                                  cfg.data['frame_stride'],
                                  cfg.data['img_size'],
                                  train)
    return SSv2Dataset(cfg.data['root'],
                       cfg.data['train_list'] if train else cfg.data['val_list'],
                       cfg.data['frames_per_clip'],
                       cfg.data['frame_stride'],
                       cfg.data['img_size'],
                       train)


def build_loader(dataset, batch_size, train=True, num_workers=8):
    sampler = DistributedSampler(dataset, shuffle=train) if dist.is_initialized() else None
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=(sampler is None and train),
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=train)
    return loader, sampler


def train_one_epoch(model, loader, optimizer, scaler, device, clip_grad=5.0, cfg=None, logger=None, epoch=0):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, acc1s, acc5s, losses = 0, 0.0, 0.0, 0.0
    center = CenterLoss(cfg['num_classes'], model.module.classifier.fc1.out_features).to(device) \
             if cfg and cfg.get('center_loss', False) else None

    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            output = model(clips)
            logits, feats, seq_embed = output if isinstance(output, tuple) else (output, None, None)
            loss = loss_fn(logits, labels)
            if center is not None:
                loss = loss + cfg.get('lambda_center', 0.1) * center(feats, labels)
            if cfg and cfg.get('temporal_loss', False) and seq_embed is not None:
                loss = loss + cfg.get('lambda_temporal', 0.05) * temporal_consistency(seq_embed)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer); scaler.update()
        acc1, acc5 = accuracy(logits, labels, topk=(1,5))
        bs = labels.size(0); total += bs
        acc1s += acc1*bs; acc5s += acc5*bs; losses += loss.item()*bs
        if logger and dist.get_rank() == 0:
            logger.log_scalars({"loss": loss.item(), "acc1": acc1, "acc5": acc5})
    if logger and dist.get_rank() == 0: logger.inc()
    return losses/total, acc1s/total, acc5s/total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, acc1s, acc5s, losses = 0, 0.0, 0.0, 0.0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        output = model(clips)
        logits = output[0] if isinstance(output, tuple) else output
        loss = loss_fn(logits, labels)
        acc1, acc5 = accuracy(logits, labels, topk=(1,5))
        bs = labels.size(0); total += bs
        acc1s += acc1*bs; acc5s += acc5*bs; losses += loss.item()*bs
    return losses/total, acc1s/total, acc5s/total


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()