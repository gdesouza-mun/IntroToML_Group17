from fp_data import get_m2020, AI4Mars_DataSet, glb

import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassRecall

import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

'''
This files contain classes to use as loss functions
'''

#=============================================
# LOSS FUNCTIONS
#=============================================

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

    DOES NOT WORK currently with the training loop, it's here give log_cosh_dice_loss needs it
    '''
    def __init__(self, num_classes=5, bg_class=255, smooth=1,
                 relative_abundance=glb.relative_abundance):
        super(dice_loss, self).__init__()
        self.num_classes = num_classes
        self.bg_class=bg_class
        self.num_true_classes = num_classes-1
        self.smooth = 0.01
        if relative_abundance is not None:
            weights_tensor = torch.tensor([1/a for a in glb.relative_abundance]).float()
            self.register_buffer('weights', weights_tensor)

        else:
            self.weights = torch.ones(len(relative_abundance))

    def process_masks(self, masks):
        '''
        I tried and failed to process the masks in a way that won't break my training loop,
        please ignore
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
        clean_outputs = outputs.clone()
        clean_outputs = outputs[:,:self.num_true_classes,:,:]

        return clean_outputs


    def forward(self, outputs, masks):
        final_masks = self.process_masks(masks)
        final_outputs = self.process_outputs(outputs)

        #Given the clean outputs, I call DiceScore on them
        # 2. IMPORTANT: Convert logits to probabilities using Softmax
        # This keeps the gradient 'alive'
        preds = torch.softmax(final_outputs, dim=1)

        weights = self.weights.to(preds.device)
        smooth = self.smooth
        weighted_dice=0
        for c in range(self.num_true_classes):
            pred_c = preds[:,c]
            mask_c = final_masks[:,c]

            intersection = (pred_c*mask_c).sum(dim=(1,2))
            union = pred_c.sum(dim=(1,2)) + mask_c.sum(dim=(1,2))
            dice_c = (2.*intersection+smooth)/(union+smooth)
            weighted_dice += (weights[c]*dice_c)


        return 1 - (weighted_dice.mean()/ weights.sum())

class log_cosh_dice_loss(dice_loss):
    '''
    Takes log ( cosh ( dice loss))
    '''
    def __init__(self, num_classes=5, bg_class=255, smooth=1,
                 relative_abundance=glb.relative_abundance):
        super(log_cosh_dice_loss, self).__init__()
        self.dice_loss = dice_loss(num_classes, bg_class, smooth,
                                   relative_abundance)

    def forward(self, outputs, masks):

        loss_value = self.dice_loss(outputs, masks)

        return torch.log(torch.cosh(loss_value))


#=============================================
# ASSESSMENT FUNCTIONS
#=============================================

'''
This first three functions I used to test and learn how to correctly
implement the torchmetric assessement
'''
def get_pixel_accuracy(preds, masks, device=torch.device("cpu"),
                       assess_name=True):
    metric = MulticlassAccuracy(
        num_classes=len(glb.class_names),
        average=None,
        ignore_index=255
        ).to(device)

    acc_per_class = metric(preds, masks)

    results = {name: acc.item() for name, acc in zip(glb.class_names, acc_per_class)}
    results["mean"] = acc_per_class.mean().item()
    if assess_name:
        results["assessment"] = "accuracy"

    return results

def get_pixel_recall(preds, masks, device=torch.device("cpu"),
                       assess_name=True):

    metric = MulticlassRecall(
        num_classes=len(glb.class_names),
        average=None,
        ignore_index=255
        ).to(device)

    acc_per_class = metric(preds, masks)

    results = {name: acc.item() for name, acc in zip(glb.class_names, acc_per_class)}
    results["mean"] = acc_per_class.mean().item()
    if assess_name:
        results["assessment"] = "recall"

    return results

def get_IoU(preds, masks, device=torch.device("cpu"),
                       assess_name=True):

    metric = MulticlassJaccardIndex(
        num_classes=len(glb.class_names),
        average=None,
        ignore_index=255
        ).to(device)

    acc_per_class = metric(preds, masks)

    results = {name: acc.item() for name, acc in zip(glb.class_names, acc_per_class)}
    results["mean"] = acc_per_class.mean().item()
    if assess_name:
        results["assessment"] = "IoU"

    return results

def validate_model(model, val_loader, device=torch.device("cpu")):

    '''
    Outputs an overall assessment as a pandas dataframe, I use to generate
    a history uof the training
    '''
    num_classes = len(glb.class_names)

    acc_metric = MulticlassAccuracy(num_classes=num_classes, average=None,
                                    ignore_index=255).to(device)
    rec_metric = MulticlassRecall(num_classes=num_classes, average=None,
                                    ignore_index=255).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average=None,
                                    ignore_index=255).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images) # Shape [B, 5, H, W]
            preds = torch.argmax(outputs['out'], dim=1)

            # 2. Update metrics (this accumulates stats internally)
            acc_metric.update(preds, masks)
            rec_metric.update(preds, masks)
            iou_metric.update(preds, masks)

    # 3. Compute final values for the whole epoch
    # These return tensors of shape [num_classes]
    final_acc = acc_metric.compute()
    final_rec = rec_metric.compute()
    final_iou = iou_metric.compute()
    metrics_list = []
    for values, name in zip([final_acc, final_rec, final_iou], ["Accuracy", "Recall", "IoU"]):
        row = {glb.class_names[i]: values[i].item() for i in range(num_classes)}
        row["mean"] = values.mean().item()
        row["assessment"] = name
        metrics_list.append(row)

    acc_metric.reset()
    rec_metric.reset()
    iou_metric.reset()

    return pd.DataFrame(metrics_list).set_index("assessment")
