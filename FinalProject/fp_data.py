import os
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import random

class glb:
    m2020_nav_img_path="AI4Mars_Data/m2020_nav/img"
    m2020_nav_mask_path="AI4Mars_Data/m2020_nav/labels"

    pallet_nav={
        0 : [21, 171, 234],
        1 : [191, 21, 234],
        2 :  [234, 84, 21],
        3 : [64, 234, 21],
        255 :  [255,255, 255]
    }

    labels_nav = {
        0:{
            "mask_rgb": [0,0,0],
            "display_rgb": [21, 171, 234], #Light Blue
            "name": "soil"
        },
        1:{
            "mask_rgb": [1,1,1],
            "display_rgb": [191, 21, 234], #Purple
            "name": "bedrock"
        },
        2:{
            "mask_rgb": [2,2,2],
            "display_rgb": [234, 84, 21], #Light Orange
            "name": "sand"
        },
        3:{
            "mask_rgb": [3,3,3],
            "display_rgb": [64, 234, 21], #Green
            "name": "big rock"
        },
        255:{
            "mask_rgb": [255,255, 255],
            "display_rgb": [255,255, 255], #White
            "name": "unlabeled"
        },
    }

    norm_mean=0.5
    norm_std=0.5

#Visualizer


def tensor_to_numpy(tensor, denormalize=True, mean=glb.norm_mean, std=glb.norm_std):
    img = tensor.detach().cpu().numpy()

    img = img.transpose(1,2,0)

    if denormalize:
        img = (img*std)+mean

    img = np.clip(img, 0, 1)

    #img = (img*255).astype(np.uint8)

    return img



def mask_to_rgb(mask, pallete=glb.pallet_nav):

    h, w = mask.shape
    rgb_mask = np.zeros((h,w,3))

    for label, color in pallete.items():
        rgb_mask[mask == label] = color

    return rgb_mask


def plot_overlay_side(image, gray_mask, pallete=glb.pallet_nav):

    rgb_mask = mask_to_rgb(gray_mask, pallete)
    # 3. Superimpose the Mask
    # Create a copy and work with floats to prevent overflow during blending
    overlay = image.copy()

    # Identify non-background pixels (where the mask is active)
    #active_mask = gray_mask > 0

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


class AI4Mars_DataSet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir,
                                 self.images[index].replace(".jpeg", ".png"))

        print(mask_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask





# m2020_train_ds = AI4Mars_DataSet(glb.m2020_nav_img_path,
#                                  glb.m2020_nav_mask_path)


# idx = random.randint(0, len(m2020_train_ds)-1)
# img, msk = m2020_train_ds[idx]

# #print(np.unique(msk))
# plot_overlay_side(img, msk, alpha=0.3)

# width = 1024
# height = 1024

# base_transform = A.Compose([
#     A.Resize(height=height, width=width),
#     A.ToGray(p=1.0),
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])

# aug_transform = A.Compose([
#     A.Resize(height=height, width=width),
#     A.ToGray(p=1.0),
#     A.HorizontalFlip(p=0.5),
#     A.CoarseDropout(
#         min_holes=4,
#         max_holes=12,
#         max_height=20,
#         max_width=20,
#         min_height=10,
#         min_width=10,
#         fill_value=0,
#         fill_mask=255,
#         p=0.5

#     ),
#     A.RandomBrightnessContrast(p=0.2),
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])

# m2020_train_ds = AI4Mars_DataSet(glb.m2020_nav_img_path,
#                                  glb.m2020_nav_mask_path, transform=aug_transform)

# idx = random.randint(0, len(m2020_train_ds)-1)
# img, msk = m2020_train_ds[10]

# print(type(img))
# print(type(msk))

#print(np.unique(msk))
#plot_overlay_side(img, msk, alpha=0.3)

#img = tensor_to_numpy(img)
#print(np.unique(msk))

#plot_overlay_side(img, msk)
