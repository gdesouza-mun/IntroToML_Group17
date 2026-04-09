from fp_data import get_m2020, AI4Mars_DataSet, glb

import numpy as np
from train_resnet import get_deeplabv3

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import JaccardIndex

import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

'''
This files contain classes to use as loss functions
'''
class abundance_weighted_CE_loss(nn.Module):
    '''
    Given the relative abundance of the classes (from 0 to 1)
    It weights each class in the cross entropy loss as 1/abundance, ignoring the
    background index
    '''
    def __init__(self, relative_abundance=glb.relative_abundance, ignore_index=255):
        super(abundance_weighted_CE_loss, self).__init__()
        self.relative_abundance=relative_abundance
        self.ignore_index=ignore_index

        weights = np.zeros(len(glb.relative_abundance))

        for i in range(len(glb.relative_abundance)):
            weights[i] = 1/glb.relative_abundance[i]

        weights = np.append(weights, [0.0,]) #background weight
        self.num_classes = len(weights)
        weights_tensor = torch.from_numpy(weights)

        self.score=nn.CrossEntropyLoss(weight=weights_tensor.float(),
                                       ignore_index=self.ignore_index)

    def forward(self, outputs, masks):
        return self.score(outputs.float(), masks.long())/self.num_classes

class dice_loss(nn.Module):
    '''
    Computes dice loss
    if given average = none and abundance list it does so with
    abundance weighted mean
    '''
    def __init__(self, num_classes=5, bg_class=255, average='none',
                 relative_abundance=glb.relative_abundance):
        super(dice_loss, self).__init__()
        self.num_classes = num_classes
        self.bg_class=bg_class
        self.average=average
        self.num_true_classes = num_classes-1

        if relative_abundance is None:
            self.weights=None
        else:
            self.weights = np.zeros(len(glb.relative_abundance))
            for i in range(len(glb.relative_abundance)):
                self.weights[i] = 1/glb.relative_abundance[i]


        self.score = DiceScore(self.num_true_classes, include_background=True,
                               average=self.average)


    def process_masks(self, masks):
        '''
        This is a bit elaborate, but the DiceScore uses hot encoded tensors
        and hot encoded tensors assume your class label is smaller than the
        number of classes

        So I have to manually drop the 255 background layer to correctly call DiceScore
        '''
        #Create a new mask
        clean_mask = masks.clone()
        #Set the background to class numclasses -1
        clean_mask[masks==self.bg_class] = self.num_true_classes

        #With this I can create a num_classes layered hot encoded mask
        #Layer num_classes -1 (the last) is the background
        mask_one_hot = F.one_hot(clean_mask.long(), num_classes=self.num_classes)

        #I drop the background layer and permute the arrays to fit with the outputs
        mask_one_hot = mask_one_hot[..., :self.num_true_classes].permute(0,3,1,2).float()

        return mask_one_hot

    def process_outputs(self, outputs):
        #I manually drop the background layer
        clean_outputs = outputs[:,:self.num_true_classes,:,:]

        return clean_outputs


    def forward(self, outputs, masks):
        final_masks = self.process_masks(masks)
        final_outputs = self.process_outputs(outputs)

        #Given the clean outputs, I call DiceScore on them
        score=self.score(final_outputs, final_masks)

        #if average is none, the output is an array for the dice score
        #of each class
        if(self.average=='none'):
            #Dice Loss is 1 - Dice Score
            per_class_loss = 1 - score

            #If I have weights, I manually take the weighted mean
            if self.weights is not None:
                weighted_loss = np.zeros(len(per_class_loss))

                for i in range(len(per_class_loss)):
                    weighted_loss[i] = per_class_loss[i] * self.weights[i]

                return torch.tensor(weighted_loss.sum()/self.weights.sum()).float()

            #Otherwise I return the regular mean
            return per_class_loss.mean()

        #If other average argument, I just take the DiceLoss as 1 - DiceScore
        return (1-self.score(final_outputs, final_masks))

class log_cosh_dice_loss(dice_loss):
    '''
    Takes log ( cosh ( dice loss))
    '''
    def __init__(self, num_classes=5, bg_class=255, average='none',
                 relative_abundance=glb.relative_abundance):
        super(log_cosh_dice_loss, self).__init__()
        self.dice_loss = dice_loss(num_classes, bg_class, average,
                                   relative_abundance)

    def forward(self, outputs, masks):

        loss_value = self.dice_loss(outputs, masks)

        return torch.log(torch.cosh(loss_value))


def tester():
    img_path, mask_path = get_m2020()
    #Usual transforms for every image

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    base_transform = A.Compose([
        A.Resize(height=128, width=128),
        A.ToGray(p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])
    #Load and split the data, notice I'm not passing any transform here
    dataset_m2020 = AI4Mars_DataSet(img_path, mask_path, transform=base_transform)
    loader = DataLoader(dataset_m2020, batch_size=8)

    model, params = get_deeplabv3()
    pretrained_path = "models/pretrainv2.pth"
    saved_parameters = torch.load(pretrained_path)
    model.load_state_dict(saved_parameters['model_state_dict'])

    images, masks = next(iter(loader))

    model.eval()
    with torch.no_grad():
        outputs = model(images)

        aux_out = outputs['aux']
        main_out = outputs['out']

        print(f"output shape: {main_out.shape}")
        print(f"masks shape: {masks.shape}")
        #print(f"Model weight dtype: {next(model.parameters()).dtype}")

        WCE_loss = abundance_weighted_CE_loss()
        print(WCE_loss(main_out, masks))

        DS_loss = dice_loss(relative_abundance=None, average='macro')
        print(DS_loss(main_out, masks))

        lg_DS_loss = log_cosh_dice_loss()
        print(lg_DS_loss(main_out, masks))
