from fp_data import glb, AI4Mars_DataSet
from train_resnet import get_deeplabv3
import random
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os


def compute_class_abundance(folder_path, num_images=None):
    """
    Computes relative abundance of classes in segmentation masks.

    Args:
        folder_path (str): Path to the folder containing PNG masks.
        num_images (int, optional): Number of images to process for testing.
                                    If None, processes the entire folder.

    Returns:
        dict: {class_name: relative_abundance}
    """
    # Mapping based on your provided schema
    class_map = {
        0: 'soil',
        1: 'bedrock',
        2: 'sand',
        3: 'big rock'
    }
    ignore_val = 255

    # Initialize counts for valid classes
    total_counts = {k: 0 for k in class_map.keys()}

    # Gather file paths
    folder = Path(folder_path)
    mask_files = sorted([f for f in folder.glob('*.png')])

    # Subset for testing if requested
    if num_images:
        mask_files = mask_files[:num_images]

    if not mask_files:
        print("No PNG files found in the directory.")
        return {}

    for mask_path in mask_files:
        # Load image and convert to numpy array
        # Since RGB values are identical (0,0,0), we convert to grayscale 'L'
        with Image.open(mask_path) as img:
            mask = np.array(img.convert('L'))

        # Count occurrences of each class
        for class_val in class_map.keys():
            total_counts[class_val] += np.sum(mask == class_val)

    # Calculate total valid pixels (excluding the 255/unlabeled pixels)
    grand_total = sum(total_counts.values())

    if grand_total == 0:
        return {name: 0.0 for name in class_map.values()}

    # Convert to relative abundance dict
    abundance = {
        class_map[val]: (count / grand_total)
        for val, count in total_counts.items()
    }

    return abundance

def plot_abundance(abundance_dict):
    """
    Creates a pie chart with a clean legend containing percentages.
    """
    palette_rgb = {
        'soil': [21, 171, 234],
        'bedrock': [191, 21, 234],
        'sand': [234, 84, 21],
        'big rock': [64, 234, 21]
    }

    labels = []
    sizes = []
    colors = []
    legend_labels = []

    for class_name, value in abundance_dict.items():
        clean_name = class_name.lower().strip()

        if clean_name in palette_rgb:
            # Data for the pie
            sizes.append(value)
            colors.append(np.array(palette_rgb[clean_name]) / 255.0)

            # Formatted label for the legend: "Class: 00.0%"
            percentage = value * 100
            legend_labels.append(f"{class_name.capitalize()}: {percentage:.1f}%")

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the pie without internal labels (autopct=None)
    patches, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )

    # Add the legend to the side
    ax.legend(
        patches,
        legend_labels,
        title="Terrain Classes",
        loc="center left",
        bbox_to_anchor=(0.75, -0.25, 0.5, 1) # Positions legend outside the circle
    )

    #plt.title('Terrain Class Distribution')
    plt.axis('equal')

    # Save with tight layout to ensure the legend isn't cut off
    plt.savefig('abundance_legend_chart.png', bbox_inches='tight', dpi=300)
    print("Graph saved as 'abundance_legend_chart.png'")


def mask_to_rgb(mask, pallete=glb.pallet_nav):
    #Given a mask in grayscale, turns it into an rgb mask according to our pallete
    h, w = mask.shape
    rgb_mask = np.zeros((h,w,3))

    for label, color in pallete.items():
        rgb_mask[mask == label] = color

    return rgb_mask

def plot_mask_overlay(image, mask, alpha=0.5, color_max=255, ax=None):
    """
    Overlays mask on image, making pixels with value 255 completely transparent.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize image
    img_norm = image.astype(float) / color_max

    # 1. Prepare the mask as RGBA
    # Create an array of shape (H, W, 4)
    h, w, _ = mask.shape
    rgba_mask = np.zeros((h, w, 4))

    # Fill RGB channels
    rgba_mask[..., :3] = mask.astype(float) / color_max

    # 2. Set Alpha channel
    # Default alpha for all pixels
    rgba_mask[..., 3] = alpha

    # 3. Find 'unlabeled' pixels (where any channel is 255)
    # If your 255 is across all RGB channels:
    unlabeled_indices = np.all(mask == 255, axis=-1)

    # Set alpha to 0 for those pixels (completely transparent)
    rgba_mask[unlabeled_indices, 3] = 0

    # Plot
    ax.imshow(img_norm)
    ax.imshow(rgba_mask) # Matplotlib handles the internal per-pixel alpha automatically
    ax.axis('off')

    return ax

# --- How to use this in a Grid ---
def create_grid_example(image_list, mask_list):
    num_imgs = len(image_list)
    fig, axes = plt.subplots(1, num_imgs, figsize=(num_imgs * 4, 4))

    # If there's only one image, axes isn't a list, so we wrap it
    if num_imgs == 1:
        axes = [axes]

    for i in range(num_imgs):
        plot_mask_overlay(
            image_list[i],
            mask_list[i],
            alpha=0.4,
            color_max=255,
            ax=axes[i]
        )

    plt.tight_layout()
    plt.show()



import albumentations as A
from albumentations.pytorch import ToTensorV2

height=512
width=512
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

#Usual transforms for every image
base_transform = A.Compose([
    A.Resize(height=height, width=width),
    A.ToGray(p=1.0),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2()
])

m2020_dataset = AI4Mars_DataSet(glb.m2020_nav_img_path, glb.m2020_nav_mask_path, transform=base_transform)

device = torch.device("cpu")

to_load="PT_LCDL"
model = get_deeplabv3
saved_parameters = torch.load(f"final_models/{to_load}_train512.pth", map_location=device)
best_epoch = saved_parameters["epoch"]
model.load_state_dict(saved_parameters['model_state_dict'])





#load model

# #m2020_dataset = AI4Mars_DataSet(glb.m2020_nav_img_path, glb.m2020_nav_mask_path)
# img, mask = m2020_dataset[15]

# print(torch.max(img))
#print(np.max(img))

# mask = mask_to_rgb(mask)

# ax = plot_mask_overlay(img, mask)
# plt.show()


# abd = compute_class_abundance(glb.m2020_nav_mask_path)

# print(abd)

# plot_abundance(abd)

