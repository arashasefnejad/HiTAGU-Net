import torch, argparse
from torch.utils.data import DataLoader
from hitagu_net.engine.utils import load_config
from hitagu_net.engine.train import build_dataset, evaluate
from hitagu_net.models import HiTAGUNet

def _load_weights(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            sd = obj['model']
        elif 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            sd = obj['state_dict']
        else:
            sd = obj
    else:
        sd = obj
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    model = HiTAGUNet(
        num_classes=cfg.model['num_classes'],
        feature_dim=cfg.model['feature_dim'],
        hidden_dim=cfg.model['hidden_dim'],
        attn_dim=cfg.model['attn_dim'],
        drop=cfg.model['drop']
    ).to(device)
    model = _load_weights(model, args.checkpoint, device)

    val_ds = build_dataset(cfg, train=False)
    bs = cfg.train.get('batch_size', 16)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)

    loss, a1, a5 = evaluate(model, val_loader, device)
    print(f"Eval — loss {loss:.4f} | top1 {a1:.2f} | top5 {a5:.2f}")

if __name__ == '__main__':
    main()