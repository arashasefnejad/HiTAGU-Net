import os
import csv
import random
import warnings
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import build_transforms


class Kinetics700Dataset(Dataset):
    
    def __init__(
        self,
        root: str,
        csv_file: str,
        frames_per_clip: int = 32,
        frame_stride: int = 2,
        img_size: int = 224,
        train: bool = True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pad_if_short: bool = True,
        fail_on_missing: bool = False,
    ):
        self.root = root
        self.items: List[Tuple[str, int, int]] = []
        self.fpc = int(frames_per_clip)
        self.stride = int(frame_stride)
        self.tx = build_transforms(img_size, is_train=train, mean=mean, std=std)
        self.train = train
        self.pad_if_short = pad_if_short
        self.fail_on_missing = fail_on_missing

        csv_path = os.path.join(root, csv_file)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    label = int(row[1])
                    num_frames = int(row[2])
                except ValueError:
                    continue
                rel = row[0].strip()
                if not rel:
                    continue
                self.items.append((rel, label, num_frames))

        if len(self.items) == 0:
            raise RuntimeError(f"No valid entries parsed from CSV: {csv_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _needed_span(self) -> int:
        return max(1, (self.fpc - 1) * self.stride + 1)

    def _sample_indices_train(self, nf: int) -> List[int]:
       
        need = self._needed_span()
        if nf >= need:
            max_start = nf - need + 1
            start = random.randint(1, max_start)
            idxs = [start + i * self.stride for i in range(self.fpc)]
        else:
            if self.pad_if_short:
                
                step = max(nf / float(self.fpc), 1.0)
                idxs = [min(int(1 + round(i * step)), nf) for i in range(self.fpc)]
            else:
                raise ValueError(
                    f"Video too short (nf={nf}) for required span={need} with T={self.fpc}, stride={self.stride}"
                )
        return idxs

    def _sample_indices_eval(self, nf: int) -> List[int]:
      
        need = self._needed_span()
        if nf >= need:
        
            start = max(1, (nf - need) // 2 + 1)
            idxs = [start + i * self.stride for i in range(self.fpc)]
        else:
            if self.pad_if_short:
               
                step = max(nf / float(self.fpc), 1.0)
                idxs = [min(int(1 + round(i * step)), nf) for i in range(self.fpc)]
            else:
                raise ValueError(
                    f"Video too short (nf={nf}) for required span={need} with T={self.fpc}, stride={self.stride}"
                )
        return idxs

    def _open_frame(self, folder: str, i: int, nf: int) -> Image.Image:
        
        def path_for(k: int) -> str:
            return os.path.join(folder, f"{k:06d}.jpg")

        path = path_for(i)
        try:
            with Image.open(path) as im:
                return im.convert("RGB").copy()
        except FileNotFoundError:
            if self.fail_on_missing:
                raise
           
            j = max(1, min(i, nf))
            alt = path_for(j)
            if alt != path:
                try:
                    with Image.open(alt) as im:
                        warnings.warn(f"Missing frame {path}; clamped to {alt}")
                        return im.convert("RGB").copy()
                except FileNotFoundError:
                    pass
       
            for delta in range(1, min(5, nf) + 1):
                for k in (i - delta, i + delta):
                    if 1 <= k <= nf:
                        try:
                            with Image.open(path_for(k)) as im:
                                warnings.warn(f"Missing frame {path}; fallback to {k:06d}.jpg")
                                return im.convert("RGB").copy()
                        except FileNotFoundError:
                            continue
            raise FileNotFoundError(f"Could not open frame {path} or any nearby fallback.")

    def __getitem__(self, idx: int):
        rel, label, nf = self.items[idx]
        folder = os.path.join(self.root, rel)

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Frames folder not found: {folder}")

        if self.train:
            idxs = self._sample_indices_train(nf)
        else:
            idxs = self._sample_indices_eval(nf)

        frames = []
        for i in idxs:
            img = self._open_frame(folder, int(i), nf)
            frames.append(self.tx(img))  # transforms must return CHW tensor

        clip = torch.stack(frames, dim=0)  # [T, C, H, W]
        return clip, label