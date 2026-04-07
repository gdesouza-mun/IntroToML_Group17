from fp_data import *

from train_resnet import get_deeplabv3, loss_criterion
from torch.utils.data import Dataset, DataLoader, random_split


import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import JaccardIndex


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import argparse


def pretrain_model(model, params, train_loader, device, epochs=15, save_name="pretrain.pth"):

    model.to(device)
    optimizer = torch.optim.Adam(params)

    criterion = loss_criterion

    best_loss = 0

    for epoch in range(epochs):
        print(f"=============== Epoch {epoch} ============")
        model.train()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, accumulation_steps=accumulation_steps)

        print(f"Pretrain loss =  {train_loss:.4f} in epoch {epoch}")
        if epoch==0 or train_loss<best_loss:
            best_loss=train_loss

            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }

            torch.save(checkpoint, save_name)
            print(f"  *** New Best Model Saved! (Loss: {best_loss:.4f}) ***")

#HYPERPARAMETERS

size_1 = 128
size_2 = 256

batch_size=4
accumulation_steps=8

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

classifier_lr = 5e-4
backbone_lr=1e-5

aug_transform_1 = A.Compose([
    A.Resize(height=size_1, width=size_1),
    A.ToGray(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CoarseDropout(
        num_holes_range=(8,16),
        hole_height_range=(10,20),
        hole_width_range=(10,20),
        fill=0,
        fill_mask=255,
        p=0.5
    ),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

aug_transform_2 = A.Compose([
    A.Resize(height=size_2, width=size_2),
    A.ToGray(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CoarseDropout(
        num_holes_range=(8,16),
        hole_height_range=(10,20),
        hole_width_range=(10,20),
        fill=0,
        fill_mask=255,
        p=0.5
    ),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
    parser = argparse.ArgumentParser(description="Is Slurm")

    parser.add_argument("--SLURM", action="store_true", help="Use SLURM Temporary Folder")
    args = parser.parse_args()

    img1_path, msk1_path = get_msl_test(IS_SLURM=args.SLURM)
    img2_path, msk2_path = get_msl_train(IS_SLURM=args.SLURM)

    dataset_pretrain1 = AI4Mars_DataSet(img1_path, msk1_path, aug_transform_1)
    dataset_pretrain2 = AI4Mars_DataSet(img2_path, msk2_path, aug_transform_2)

    model, params = get_deeplabv3(num_classes=5, classifier_lr=1e-3, backbone_lr=backbone_lr)

    loader1 = DataLoader(dataset_pretrain1, batch_size=batch_size, shuffle=True,
                         pin_memory=True)

    loader2 = DataLoader(dataset_pretrain2, batch_size=batch_size, shuffle=True,
                         pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_model(model, params, loader1, device, epochs=30, save_name="pretrainv1.pth")
    pretrain_model(model, params, loader2, device, epochs=30, save_name="pretrainv2.pth")
