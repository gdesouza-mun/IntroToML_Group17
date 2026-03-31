from fp_data import *

from train_resnet import get_deeplabv3, loss_criterion

#HYPERPARAMETERS

size_1 = 128
size_2 = 256

batch_size=8
accumulation_steps=8

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

classifier_lr = 1e-3
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

    dataset_pretrain1 = AI4Mars_DataSet(img1_path, msk1_path)
    model, params = get_deeplabv3(classifier_lr=1e-3, backbone_lr=backbone_lr)





    #img2,msk2 = get_msl_train(IS_SLURM=args.SLURM)
