import torch
from hitagu_net.models import HiTAGUNet

def test_forward_smoke_cpu():
    model = HiTAGUNet(num_classes=10, pretrained_backbone=False)
    model.eval()
    x = torch.randn(1, 4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 10)

def test_forward_smoke_small():
    model = HiTAGUNet(num_classes=5, pretrained_backbone=False)
    model.eval()
    x = torch.randn(1, 2, 3, 112, 112)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 5)