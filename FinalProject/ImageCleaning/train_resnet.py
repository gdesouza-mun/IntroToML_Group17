from fp_data import *


from torch.utils.data import Dataset, DataLoader, random_split
import torch

#HYPERPARAMETERS
width = 1024
height = 1024


#full_data

base_transform = A.Compose([
    A.Resize(height=height, width=width),
    A.ToGray(p=1.0),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

    # A.CoarseDropout(
    #     min_holes=4,
    #     max_holes=12,
    #     max_height=20,
    #     max_width=20,
    #     min_height=10,
    #     min_width=10,
    #     fill_value=0,
    #     fill_mask=255,
    #     p=0.5

    # ),

aug_transform = A.Compose([
    A.Resize(height=height, width=width),
    A.ToGray(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])


dataset_m2020 = AI4Mars_DataSet(glb.m2020_nav_img_path, glb.m2020_nav_mask_path)

train_size = int(0.9*len(dataset_m2020))
val_size = len(dataset_m2020) - train_size

train_subset, val_subset = random_split(dataset_m2020, [train_size, val_size],
                                        generator = torch.Generator().manual_seed(17))

train_m2020 = AI4Mars_SubSet(train_subset, transform = aug_transform)
val_m2020 = AI4Mars_SubSet(val_subset, transform = base_transform)
img, msk = train_m2020[220]


print(type(img))
print(np.unique(msk))

img = tensor_to_numpy(img)
#msk = tensor_to_numpy(msk)

plot_overlay_side(img, msk)
