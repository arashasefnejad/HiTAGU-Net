import torch, argparse, os, numpy as np
from hitagu_net.engine.utils import load_config
from hitagu_net.models import HiTAGUNet
from PIL import Image
import torchvision.transforms as T
def five_crops(img, size=224):
    w,h = img.size; s=size
    return [img.crop((0,0,s,s)), img.crop((w-s,0,w,s)), img.crop((0,h-s,s,h)), img.crop((w-s,h-s,w,h)), img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True); ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--video_folder', type=str, required=True); ap.add_argument('--num_clips', type=int, default=10)
    args = ap.parse_args(); cfg = load_config(args.config); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HiTAGUNet(num_classes=cfg.model['num_classes'], feature_dim=cfg.model['feature_dim'], hidden_dim=cfg.model['hidden_dim'], attn_dim=cfg.model['attn_dim'], drop=cfg.model['drop']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device); model.load_state_dict(ckpt['model'], strict=False); model.eval()
    frames = sorted([f for f in os.listdir(args.video_folder) if f.endswith('.jpg')]); stride = max(1, len(frames)//(args.num_clips*cfg.data['frames_per_clip']))
    transform = T.Compose([T.Resize(256), T.ToTensor(), T.Normalize(cfg.data['mean'], cfg.data['std'])])
    logits_all=[]
    with torch.no_grad():
        for c in range(args.num_clips):
            clip_idxs = [min(c*stride + i*stride, len(frames)-1) for i in range(cfg.data['frames_per_clip'])]
            imgs = [Image.open(os.path.join(args.video_folder, frames[i])).convert('RGB') for i in clip_idxs]
            avg_frames = [torch.stack([transform(cr) for cr in five_crops(img, 224)], dim=0).mean(dim=0) for img in imgs]
            clip = torch.stack(avg_frames, dim=0).unsqueeze(0).to(device)
            out = model(clip); if isinstance(out, tuple): out = out[0]
            logits_all.append(out.softmax(dim=-1).cpu().numpy())
    probs = np.mean(np.stack(logits_all, axis=0), axis=0)[0]; pred = int(np.argmax(probs))
    print('Prediction:', pred, 'Prob:', float(probs[pred]))
if __name__ == '__main__': main()
