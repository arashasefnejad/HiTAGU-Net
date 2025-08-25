import torch, os, argparse
from hitagu_net.engine.utils import load_config, set_seed
from hitagu_net.engine.train import build_dataset, train_one_epoch, evaluate
from hitagu_net.models import HiTAGUNet
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda.amp import GradScaler

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=str, required=True); args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = HiTAGUNet(num_classes=cfg.model['num_classes'], feature_dim=cfg.model['feature_dim'],
                      hidden_dim=cfg.model['hidden_dim'], attn_dim=cfg.model['attn_dim'], drop=cfg.model['drop']).to(device)
    train_ds = build_dataset(cfg, train=True); val_ds = build_dataset(cfg, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.train['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.train['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    opt = SGD(model.parameters(), lr=cfg.train['lr'], momentum=cfg.train['momentum'], weight_decay=cfg.train['weight_decay'], nesterov=cfg.train['nesterov'])
    scaler = GradScaler(enabled=cfg.amp)
    best_acc=0.0; os.makedirs(cfg.train['checkpoint_dir'], exist_ok=True)
    train_extra = {'center_loss': cfg.train.get('center_loss', False),'temporal_loss': cfg.train.get('temporal_loss', False),
                   'lambda_center': cfg.train.get('lambda_center', 0.1),'lambda_temporal': cfg.train.get('lambda_temporal', 0.05),
                   'num_classes': cfg.model['num_classes']}
    for epoch in range(cfg.train['epochs']):
        tr_loss, tr_a1, tr_a5 = train_one_epoch(model, train_loader, opt, scaler, device, cfg.train['clip_grad_norm'], cfg=train_extra)
        va_loss, va_a1, va_a5 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}: train loss {tr_loss:.4f} a1 {tr_a1:.2f} | val loss {va_loss:.4f} a1 {va_a1:.2f}")
        if va_a1 > best_acc:
            best_acc = va_a1; torch.save({'model': model.state_dict(),'acc1': best_acc}, os.path.join(cfg.train['checkpoint_dir'], 'best.pt'))
if __name__ == '__main__': main()
