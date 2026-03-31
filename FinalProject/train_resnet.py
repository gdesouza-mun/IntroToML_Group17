from fp_data import *

from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchmetrics.segmentation import DiceScore
import torch.nn.functional as F

import time

import argparse


#Here I'm training a deeplabv3 model

#Add Grad Carry over for smaller batches in practice
#Pre Train logic


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


def loss_criterion(outputs, masks):

    dice_crit = DiceScore(num_classes=4, include_background=True)

    ce_crit = nn.CrossEntropyLoss()

    clean_mask = masks.clone()
    clean_mask[masks==255] = 4
    mask_one_hot = F.one_hot(clean_mask, num_classes=5)

    mask_one_hot = mask_one_hot[..., :4].permute(0,3,1,2).float()
    clean_outputs = outputs[:,:4,:,:]

    dice_loss = dice_crit(clean_outputs, mask_one_hot)
    ce_loss = ce_crit(clean_outputs, mask_one_hot)

    return dice_loss + ce_loss

def train_model(model, params, train_loader, val_loader, device, epochs=5,
                save=False, save_start=10, save_name="best_model.pth"):
    #Executes the training loop

    model.to(device)
    optimizer = torch.optim.Adam(params)
    #criterion = nn.CrossEntropyLoss(ignore_index=255)
    #criterion = DiceScore(num_classes=5, include_background=True)
    criterion = loss_criterion
    #miou_metric = MeanIoU(num_classes=5, per_class=True).to(device)

    best_miou=0.0
    print(f"Starting train on {device} for {epochs} epochs")
    for epoch in range(epochs):
        start_time = time.perf_counter()
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, accumulation_steps=accumulation_steps)
        end_time = time.perf_counter()
        duration = end_time - start_time

        # Validation
        model.eval()

        iou_per_class, val_loss = validation_metrics(model, val_loader,criterion,device)
        class_names = ["Soil", "Bedrock", "Sand", "Big Rock", "Null"]

        print(f"\n" + "="*40)
        print(f"Training duration: {duration:.0f} seconds:")
        print(f"Train Loss in {epoch} epoch: {train_loss}")
        print(f"Validation Loss in {epoch} epoch: {val_loss}")

        for name, score in zip(class_names, iou_per_class):
            print(f"{name} IoU: {score.item():.4f}")

        current_miou=iou_per_class[:4].mean().item()
        print(f"Mean IoU: {current_miou:.4f}")

        # Checkpoint
        if current_miou >= best_miou:
            best_miou=current_miou
            if save and epoch >= save_start:
                checkpoint = {
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': current_miou
                }

                torch.save(checkpoint, save_name)
                print(f"  *** New Best Model Saved! (mIoU: {best_miou:.4f}) ***")

    print(f"Training finished. Best mIoU achieved: {best_miou:.4f}")



#HYPERPARAMETERS
width = 128
height = 128
train_split = 0.9
batch_size = 8
accumulation_steps = 8

#Set Normalizization According to pretrained model
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

classifier_lr = 1e-3
backbone_lr=1e-5

#Usual transforms for every image
base_transform = A.Compose([
    A.Resize(height=height, width=width),
    A.ToGray(p=1.0),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

#usual Transforms + some agumentation options
aug_transform = A.Compose([
    A.Resize(height=height, width=width),
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

    # os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
    parser = argparse.ArgumentParser(description="Is Slurm")

    parser.add_argument("--SLURM", action="store_true", help="Use SLURM Temporary Folder")

    args = parser.parse_args()

    img_path, mask_path = get_m2020(IS_SLURM=args.SLURM)

    #Load and split the data, notice I'm not passing any transform here
    dataset_m2020 = AI4Mars_DataSet(img_path, mask_path)

    #We now get the model
    model, params = get_deeplabv3(backbone_lr=backbone_lr)



    train_size = int(train_split*len(dataset_m2020))
    val_size = len(dataset_m2020) - train_size

    train_subset, val_subset = random_split(dataset_m2020, [train_size, val_size],
                                        generator = torch.Generator().manual_seed(17))

    #Given the subsets, that are just an index split, and then redefine
    #then as the subset datasets, with the appropriate transforms
    train_m2020 = AI4Mars_SubSet(train_subset, transform = aug_transform)
    val_m2020 = AI4Mars_SubSet(val_subset, transform = base_transform)

    train_m2020_loader = DataLoader(train_m2020, batch_size=batch_size, shuffle=True,
                                    pin_memory=True)
    val_m2020_loader = DataLoader(val_m2020, batch_size=batch_size, shuffle=False,
                                  pin_memory=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(model, params, train_m2020_loader, val_m2020_loader, device, epochs=5,
                save=True, save_start=2, save_name="best_modelv1_512.pth")
