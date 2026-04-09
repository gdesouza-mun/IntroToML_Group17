from fp_data import *

from train_resnet import get_deeplabv3, loss_criterion
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset


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


def pretrain_model(model, params, train_loader, device, epochs=15,
                   save_name="pretrain.pth"):

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
        if (epoch==0 or train_loss<best_loss) and save_name:
            best_loss=train_loss

            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }

            torch.save(checkpoint, save_name)
            print(f"  *** New Best Model Saved! (Loss: {best_loss:.4f}) ***")





if __name__ == "__main__":
    '''
    Arguments
    --SIZE:<Num> Sets image size
    --BATCH:<Num> sets batch size
    --ACC:<Num> Sets Accumulation steps
    --LOAD: path to model to load (with pth)
    --SAVE: saves model to <SAVE>.pth and writes traim history on <SAVE>_hist.csv
    --SLURM uses slurm data loading set up to run on server
    '''
    # os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"

    parser = argparse.ArgumentParser(description="Model Training Set up")

    # Define arguments based on your specifications
    parser.add_argument("--SIZE", type=int, default=128, help="Sets image size")
    parser.add_argument("--BATCH", type=int, default=8, help="Sets batch size")
    parser.add_argument("--ACC", type=int, default=8, help="Sets Accumulation steps")
    parser.add_argument("--EPOCHS", type=int, default=15, help="Sets Training epochs steps")
    parser.add_argument("--LOAD", type=str, default=None,
                        help="Path to model to load (ending in .pth)")
    parser.add_argument("--SAVE", type=str, default=None,
                        help="Prefix for saving .pth and _hist.csv")
    parser.add_argument("--SLURM", action="store_true", help="Use SLURM Temporary Folder")
    args = parser.parse_args()

    img1_path, msk1_path = get_msl_test(IS_SLURM=args.SLURM)
    img2_path, msk2_path = get_msl_train(IS_SLURM=args.SLURM)
    load_path = args.LOAD
    save_prefix = args.SAVE

    #HYPERPARAMETERS
    img_size = args.SIZE
    batch_size = args.BATCH
    accumulation_steps = args.ACC
    epochs = args.EPOCHS

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    classifier_lr = 5e-4
    backbone_lr=1e-5

    aug_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.ToGray(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, interpolation=1, border_mode=0, fill=0, fill_mask=255),
        A.RandomBrightnessContrast(p=0.2),
        A.CoarseDropout(
            num_holes_range=(8,16),
            hole_height_range=(10,20),
            hole_width_range=(10,20),
            fill=0,
            fill_mask=255,
            p=0.5
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

    dataset_pretrain1 = AI4Mars_DataSet(img1_path, msk1_path, aug_transform)
    dataset_pretrain2 = AI4Mars_DataSet(img2_path, msk2_path, aug_transform)
    combined_dataset = ConcatDataset([dataset_pretrain1, dataset_pretrain2])

    model, params = get_deeplabv3(num_classes=5,
                                  classifier_lr=classifier_lr, backbone_lr=backbone_lr)
    if load_path:
        saved_parameters=torch.load(load_path)
        model.load_state_dict(saved_parameters['model_state_dict'])

    sampler = get_sampler(combined_dataset, weight=10.0)
    loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler,
                         pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_model(model, params, loader, device, epochs=epochs,
                   save_name=f"{save_prefix}.pth")
