import os
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import JaccardIndex


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


import random


# ==================================================================
# PATHS, LABEL DEFINITIONS AND VISUZALITION
# ==================================================================
class glb:
    '''
    This class is just a place to leave the path to the datasets
    and a stanard way to set the coloring for the display of the masks
    as well as classe names
    '''

    #We have two datasets, msl and m2020

    #Here is m2020 path
    m2020_nav_img_path="AI4Mars_Data/m2020_nav/img"
    m2020_nav_mask_path="AI4Mars_Data/m2020_nav/labels"

    msl_nav_img_path_train="AI4Mars_Data/msl_nav/img_train"
    msl_nav_mask_path_train="AI4Mars_Data/msl_nav/labels_train"

    msl_nav_img_path_test="AI4Mars_Data/msl_nav/img_test"
    msl_nav_mask_path_test="AI4Mars_Data/msl_nav/labels_test"


    #and here isn't msl path - I'll evetually add it
    #My plan is to pre train our models on the msl dataset
    #and see if that helps

    #Standard pallete for the mask used in articles from the lit review
    pallet_nav={
        0 : [21, 171, 234],
        1 : [191, 21, 234],
        2 :  [234, 84, 21],
        3 : [64, 234, 21],
        255 :  [255,255, 255]
    }

    #Dic of Dic with labels and names
    labels_nav = {
        0:{
            "mask_rgb": [0,0,0],
            "display_rgb": pallet_nav[0], #Light Blue
            "name": "soil"
        },
        1:{
            "mask_rgb": [1,1,1],
            "display_rgb": pallet_nav[1], #Purple
            "name": "bedrock"
        },
        2:{
            "mask_rgb": [2,2,2],
            "display_rgb": pallet_nav[2], #Light Orange
            "name": "sand"
        },
        3:{
            "mask_rgb": [3,3,3],
            "display_rgb": pallet_nav[3], #Green
            "name": "big rock"
        },
        255:{
            "mask_rgb": [255,255, 255],
            "display_rgb": pallet_nav[255], #White
            "name": "unlabeled"
        },
    }

def get_m2020(IS_SLURM=False):
    """
    Adjusts M2020 navigation and mask paths based on SLURM environment
    that I use to run the training jobs
    """
    # Define your base paths (assuming glb is accessible)
    nav_path = glb.m2020_nav_img_path
    mask_path = glb.m2020_nav_mask_path

    if IS_SLURM:
        # Get the temporary directory from environment variables
        slurm_tmp = os.environ.get('SLURM_TMPDIR', '')

        # Join the temp directory with the existing paths
        nav_path = os.path.join(slurm_tmp, nav_path)
        mask_path = os.path.join(slurm_tmp, mask_path)

    return nav_path, mask_path

def get_msl_train(IS_SLURM=False):
    """
    Adjusts M2020 navigation and mask paths based on SLURM environment.
    """
    # Define your base paths (assuming glb is accessible)
    nav_path = glb.msl_nav_img_path_train
    mask_path = glb.msl_nav_mask_path_train

    if IS_SLURM:
        # Get the temporary directory from environment variables
        slurm_tmp = os.environ.get('SLURM_TMPDIR', '')

        # Join the temp directory with the existing paths
        nav_path = os.path.join(slurm_tmp, nav_path)
        mask_path = os.path.join(slurm_tmp, mask_path)

    return nav_path, mask_path

def get_msl_test(IS_SLURM=False):
    """
    Adjusts M2020 navigation and mask paths based on SLURM environment.
    """
    # Define your base paths (assuming glb is accessible)
    nav_path = glb.msl_nav_img_path_test
    mask_path = glb.msl_nav_mask_path_test

    if IS_SLURM:
        # Get the temporary directory from environment variables
        slurm_tmp = os.environ.get('SLURM_TMPDIR', '')

        # Join the temp directory with the existing paths
        nav_path = os.path.join(slurm_tmp, nav_path)
        mask_path = os.path.join(slurm_tmp, mask_path)

    return nav_path, mask_path

def tensor_to_numpy(tensor, denormalize=True, mean=0.5, std=0.5):
    #Converts a tensor to NP array, useful sometimes for visualization
    #Might not work with mean defined per channel
    #used this mostly to test other stuff
    img = tensor.detach().cpu().numpy()

    img = img.transpose(1,2,0)

    if denormalize:
        img = (img*std)+mean

    img = np.clip(img, 0, 1)

    #img = (img*255).astype(np.uint8)

    return img



def mask_to_rgb(mask, pallete=glb.pallet_nav):
    #Given a mask in grayscale, turns it into an rgb mask according to our pallete
    h, w = mask.shape
    rgb_mask = np.zeros((h,w,3))

    for label, color in pallete.items():
        rgb_mask[mask == label] = color

    return rgb_mask


def plot_overlay_side(image, gray_mask, pallete=glb.pallet_nav):
    #Plots an image an a mask side by side
    rgb_mask = mask_to_rgb(gray_mask, pallete)

    overlay = image.copy()

    overlay = rgb_mask

    # 4. Plot Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title(f"Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# ==================================================================
# DATASETS
# ==================================================================
class AI4Mars_DataSet(Dataset):
    '''
    This is our 'global' dataset, it takes the directory files
    image_dir -> Path to images
    mask_dir -> Path to masks
    transforms -> Transforms to apply for images
    '''
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        #Figures image path, and gets mask of same name but .png extension
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir,
                                 self.images[index].replace(".jpeg", ".png"))

        #loads image using PIL
        image_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)

        #Some masks aren't the right size
        #AI4Mars says to center crop the masks if that's the case
        #so we do that
        if image_pil.size != mask_pil.size:
            target_size = (image_pil.size[1], image_pil.size[0])
            mask_pil = TF.center_crop(mask_pil, target_size)

        #Conver images to numpy
        image = np.array(image_pil)
        #with masks in a single grayscale channel
        mask = np.array(mask_pil.convert("L"))

        #If transfroms apply them
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        #Returns final image & Mask
        return image, mask

class AI4Mars_SubSet(Dataset):
    '''
    Since the m2020 dataset is all in a single folder, we can't pass transforms
    when loading the data with AI4Mars_DataSet
    So this dataset serves to split a dataset with different transforms for different splits
    You can find this implemented in the train_resnet.py

    subset-> Takes a subset from random_split method or similar
    transforms -> Transforms to apply from image
    '''
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, mask = self.subset.dataset[self.subset.indices[index]]
        #Since we get our image and mask through AI4Mars_DataSet
        #We don't have to scale the image again

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask

def get_sampler(subset, weight=5.0):
    minority_class = 3
    sample_weights = []

    for idx in range(len(subset)):
        _, mask = subset[idx]

        if (mask == minority_class).any():
            sample_weights.append(weight)
        else:
            sample_weights.append(1.0)

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples = len(sample_weights),
                                    replacement=True)
    return sampler


# ==================================================================
# ASSESSMENT AND TRAINING
# ==================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    accumulation_steps=1):
    '''
    Runs one epoch of training given a dataloader and a model, returning
    the training loss for one epoch
    '''
    model.train()
    running_loss = 0.0
    block=(device == torch.device("cuda"))
    optimizer.zero_grad()
    for i, (images, masks) in enumerate(dataloader):

        images = images.to(device, non_blocking=block)
        masks = masks.to(device, non_blocking=True).long() # Masks must be Long integers
        #print(images.shape)
        outputs = model(images)

        loss_main = criterion(outputs['out'], masks)
        # Loss from auxiliary classifier (weighted 40% usually)
        loss_aux = criterion(outputs['aux'], masks)

        total_loss = loss_main + 0.4 * loss_aux

        total_loss.backward()

        if (i+1)%accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad

        running_loss += total_loss.item()

    return running_loss / len(dataloader)

def validation_metrics(model, dataloader, criterion, device, num_classes=5):
    '''
    Since we might try more than one model, this is returns consistent validations
    This returns the IoU per classe, plus validation loss for the criterion of choice
    '''
    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes,
                           ignore_index=255, average='none').to(device)
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)['out'] # Shape: [B, 5, H, W]
            preds = torch.argmax(outputs, dim=1) # Shape: [B, H, W]

            # Update the metric state (accumulates intersection/union)
            jaccard.update(preds, masks)
            val_loss+=criterion(outputs, masks)


    # Compute final results
    # iou_per_class will be a tensor of 5 values
    iou_per_class = jaccard.compute()
    jaccard.reset()
    # Reset for next time

    return iou_per_class, val_loss/len(dataloader)


# ==================================================================
# TEST FUNCTIONS
# ==================================================================
def verify_dataset_integrity(dataset, minus_shape=-1):
    #USED to check if the center cropping was working
    print(f"Checking {len(dataset)} samples in the dataset...")

    mismatch_found = False

    # Iterate through the entire dataset
    for i in range(len(dataset)):
        image, label = dataset[i]

        if image.shape[:minus_shape] != label.shape:
            print(f"\n❌ Size Mismatch at index {i}!")
            print(f"Image size: {image.shape} | Label size: {label.shape}")
            mismatch_found = True

    if not mismatch_found:
        print("\n✅ Sanity Check Passed! All images and masks match perfectly.")
    else:
        print("\n⚠️ Sanity Check Failed. Review the errors above.")
