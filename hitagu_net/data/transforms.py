import torchvision.transforms as T
def build_transforms(img_size=224, is_train=True, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    if is_train:
        return T.Compose([T.Resize(256), T.RandomResizedCrop(img_size, scale=(0.8,1.0)), T.RandomHorizontalFlip(0.5),
                          T.ColorJitter(0.2,0.2,0.2,0.1), T.ToTensor(), T.Normalize(mean, std)])
    else:
        return T.Compose([T.Resize(256), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean, std)])
