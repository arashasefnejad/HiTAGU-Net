# scripts/convert_pretrained.py
# Map/clean state_dict keys for compatibility if checkpoints were trained under DataParallel/DDP, etc.
import torch, argparse

def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        out[nk] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_ckpt', required=True)
    ap.add_argument('--out_ckpt', required=True)
    args = ap.parse_args()
    sd = torch.load(args.in_ckpt, map_location='cpu')
    if 'state_dict' in sd:
        sd['state_dict'] = strip_module_prefix(sd['state_dict'])
        torch.save(sd, args.out_ckpt)
    else:
        sd = strip_module_prefix(sd)
        torch.save(sd, args.out_ckpt)
    print(f'[OK] Wrote {args.out_ckpt}')

if __name__ == '__main__':
    main()