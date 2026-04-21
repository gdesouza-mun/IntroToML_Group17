from fp_data import *
import fp_metrics

import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchmetrics.segmentation import DiceScore
import torch.nn.functional as F

import time

import argparse

def get_deeplabv3(num_classes=5, classifier_lr = 1e-3, backbone_lr=0.0):
    '''
    This gets the model given some conditions
    num_classes = 5 (4 classes + background)
    classifier_lr -> Learning rate for the last layer of the NN
    backbone_lr -> lr for the rest of the NN
    If backbone_lr=0, we freeze the backbone

    Returns the model + parameters that weren't frozen with their learning rates
    '''

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)

    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    if model.aux_classifier:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1))

    params_to_optimize = []

    if backbone_lr==0:
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        for param in model.backbone.parameters():
            param.requires_grad=True
        params_to_optimize.append({"params":model.backbone.parameters(), "lr": backbone_lr})

    for param in model.classifier.parameters():
        param.requires_grad = True
    params_to_optimize.append({"params": model.classifier.parameters(), "lr": classifier_lr})

    if model.aux_classifier:
        for param in model.aux_classifier.parameters():
            param.requires_grad = True
        params_to_optimize.append({"params": model.aux_classifier.parameters(),
                                       "lr": classifier_lr})

    return model, params_to_optimize


def train_model(model,
                criterion, optimizer,
                train_loader, val_loader,
                device, epochs=5, accumulation_steps=1,
                save_start=10, save_name=None,
                last_epoch=0):
    #Executes the training loop

    model.to(device)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.9)
    #miou_metric = MeanIoU(num_classes=5, per_class=True).to(device)

    best_miou=0.0
    if save_name is not None:
        pth_save_name = f"{save_name}.pth"

    val_hist_arr = []
    train_hist_arr = []
    epochs_arr = []
    print(f"Starting train on {device} for {epochs} epochs")
    for epoch in range(1, epochs+1):
        epochs_arr.append(epoch)
        start_time = time.perf_counter()
        print(f"================ Epoch {epoch} ============== ")
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, accumulation_steps=accumulation_steps)
        scheduler.step()

        print(f"Current training loss:\t {train_loss:.4}")

        if save_name is not None:
            #I save the current best performing model as I go
            true_epoch = last_epoch+epoch
            train_score = fp_metrics.validate_model(model, train_loader, device)
            train_score["dataset"] = "train"
            train_score["epoch"] = true_epoch

            train_hist_arr.append(train_score)

            val_score = fp_metrics.validate_model(model, val_loader, device)
            val_score["dataset"] = "validation"
            val_score["epoch"] = true_epoch

            val_hist_arr.append(val_score)

            current_miou = val_score.at["IoU", "mean"]

            for name in glb.class_names:
                print(f"Current {name} IoU: \t {val_score.at['IoU', name]:.4f}")

            print(f"Mean IoU: \t {current_miou:.4f}")
            if current_miou >= best_miou:
                best_miou=current_miou
                if epoch >= save_start:
                    checkpoint = {
                        'epoch': true_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'miou': current_miou
                    }

                    torch.save(checkpoint, pth_save_name)
                    print(f"  *** New Best Model Saved! (mIoU: {best_miou:.4f}) ***")
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Time for full epoch {duration:.0f} seconds")


    if save_name is not None:
        #I also print a history of validation metrics as a function of epochs
        #So I can make some nice graphs
        val_hist_save = f"{save_name}_val.csv"
        train_hist_save = f"{save_name}_train.csv"

        if len(val_hist_arr)>0:
            val_df = pd.concat(val_hist_arr, keys=epochs_arr)
            val_df.to_csv(val_hist_save)

        if len(train_hist_arr)>0:
            train_df = pd.concat(train_hist_arr, keys=epochs_arr)
            train_df.to_csv(train_hist_save)
            print(f"Training history saved to {train_hist_save} and {val_hist_save}")

    print(f"Training finished. Best mIoU achieved: {best_miou:.4f}")


if __name__ == "__main__":

    '''
    I pass a bunch of my hyperparameters from the command line
    so I can control the script from inside the scripts I use to call the
    jobs.
    '''

    parser = argparse.ArgumentParser(description="Is Slurm")

    parser.add_argument("--SIZE", type=int, default=128, help="Sets image size")
    parser.add_argument("--BATCH", type=int, default=8, help="Sets batch size")
    parser.add_argument("--ACC", type=int, default=8, help="Sets Accumulation steps")
    parser.add_argument("--EPOCHS", type=int, default=15, help="Sets Training epochs steps")
    parser.add_argument("--LOSS", type=str, default="CE", help="Sets loss function")
    parser.add_argument("--LOAD", type=str, default=None,
                        help="Path to model to load (ending in .pth)")
    parser.add_argument("--SAVE", type=str, default=None,
                        help="Prefix for saving .pth and _hist.csv")
    parser.add_argument("--SLURM", action="store_true", help="Use SLURM Temporary Folder")
    parser.add_argument("--FRESH", action="store_true", help="Forces starting epoch to be 0")


    args = parser.parse_args()

    img_path, mask_path = get_m2020(IS_SLURM=args.SLURM)

    #HYPERPARAMETERS
    img_size = args.SIZE
    width = img_size
    height = img_size
    train_split = 0.9
    batch_size = args.BATCH
    accumulation_steps = args.ACC
    epochs = args.EPOCHS



    #Set Normalizization According to pretrained model
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    classifier_lr = 5e-4
    backbone_lr=1e-5

    #Usual transforms for every image
    base_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.ToGray(p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

    #usual Transforms + some agumentation options
    aug_transform = A.Compose([
        A.Resize(height=height, width=width),
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
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #We now get the model
    model, params = get_deeplabv3(backbone_lr=backbone_lr)

    model.to(device)
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)

    #I used Adam for prototyping, SGD for the final runs
    #optimizer = torch.optim.Adam(params)

    load_path=args.LOAD
    last_epoch=0
    if load_path is not None:
            print(f"Loading model from {load_path:}")
            saved_parameters = torch.load(load_path, weights_only=False, map_location=device)
            model.load_state_dict(saved_parameters['model_state_dict'])
            #optimizer.load_state_dict(saved_parameters['optimizer_state_dict'])
            if not args.FRESH:
                last_epoch=saved_parameters['epoch']



    #I get my dataset
    dataset_m2020 = AI4Mars_DataSet(img_path, mask_path)

    #Split it 90/10 into train/val
    train_size = int(train_split*len(dataset_m2020))
    val_size = len(dataset_m2020) - train_size
    train_subset, val_subset = random_split(dataset_m2020, [train_size, val_size],
                                        generator = torch.Generator().manual_seed(17))

    #Given the subsets, that are just an index split, and then redefine
    #then as the subset datasets, with the appropriate transforms
    train_m2020 = AI4Mars_SubSet(train_subset, transform = aug_transform)
    val_m2020 = AI4Mars_SubSet(val_subset, transform = base_transform)

    #Get my weighted sampler
    sampler = get_sampler(train_m2020, 15.0)
    train_m2020_loader = DataLoader(train_m2020, batch_size=batch_size, sampler=sampler)
    val_m2020_loader = DataLoader(val_m2020, batch_size=batch_size, shuffle=False,
                                  pin_memory=True)


    #I also choose the loss from the command line so
    # I can train different models from this single script
    loss_flag=args.LOSS

    if loss_flag=="CE":
        criterion = fp_metrics.abundance_weighted_CE_loss()
    elif loss_flag=="DL":
        criterion = fp_metrics.dice_loss()
    elif loss_flag=="LCDL":
        criterion = fp_metrics.log_cosh_dice_loss(smooth=1e-6)
    else:
        print("Unkown loss flag, setting loss to cross entropy")
        criterion = abundance_weighted_CE_loss()

    save_path=args.SAVE
    #Call my training loop
    train_model(model,
                criterion, optimizer,
                train_m2020_loader, val_m2020_loader, device,
                epochs=epochs, accumulation_steps=accumulation_steps,
                save_start=0, save_name=save_path, last_epoch=last_epoch)
