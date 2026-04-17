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

# from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# def get_deeplabv3(num_classes=5, classifier_lr = 1e-3, backbone_lr=0.0):
#     '''
#     This gets the model given some conditions
#     num_classes = 5 (4 classes + background)
#     classifier_lr -> Learning rate for the last layer of the NN
#     backbone_lr -> lr for the rest of the NN
#     If backbone_lr=0, we freeze the backbone

#     Returns the model + parameters that weren't frozen with their learning rates
#     '''

#     weights = DeepLabV3_ResNet50_Weights.DEFAULT
#     model = deeplabv3_resnet50(weights=weights)

#     model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
#     if model.aux_classifier:
#         in_channels_aux = model.aux_classifier[4].in_channels
#         model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1))

#     params_to_optimize = []

#     if backbone_lr==0:
#         for param in model.backbone.parameters():
#             param.requires_grad = False
#     else:
#         for param in model.backbone.parameters():
#             param.requires_grad=True
#         params_to_optimize.append({"params":model.backbone.parameters(), "lr": backbone_lr})

#     for param in model.classifier.parameters():
#         param.requires_grad = True
#     params_to_optimize.append({"params": model.classifier.parameters(), "lr": classifier_lr})

#     if model.aux_classifier:
#         for param in model.aux_classifier.parameters():
#             param.requires_grad = True
#         params_to_optimize.append({"params": model.aux_classifier.parameters(),
#                                        "lr": classifier_lr})

#     return model, params_to_optimize


# def tester():
#     img_path, mask_path = get_m2020(True)
#     #Usual transforms for every image

#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]

#     base_transform = A.Compose([
#         A.Resize(height=128, width=128),
#         A.ToGray(p=1.0),
#         A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
#         ToTensorV2()
#     ])
#     #Load and split the data, notice I'm not passing any transform here
#     dataset_m2020 = AI4Mars_DataSet(img_path, mask_path, transform=base_transform)
#     loader = DataLoader(dataset_m2020, batch_size=8)

#     model, params = get_deeplabv3()
#     pretrained_path = "final_models/CE_train128.pth"
#     saved_parameters = torch.load(pretrained_path, weights_only=False)
#     model.load_state_dict(saved_parameters['model_state_dict'])

#     device = torch.device("cuda")

#     # pd1 = validate_model(model, loader, device)
#     # print(pd1.round(4))
#     # pd1["epoch"]=0
#     # pd2 = validate_model(model, loader, device)
#     # pd2["epoch"]=1
#     # print(pd.concat([pd1, pd2]).round(4))

#     images, masks = next(iter(loader))

#     model.eval()
#     with torch.no_grad():
#         outputs = model(images)

#         aux_out = outputs['aux']
#         main_out = outputs['out']

#         preds = torch.argmax(main_out, dim=1)

#         print(type(preds))
#         print(type(masks))

#         print(f"output shape: {preds.shape}")
#         print(f"masks shape: {masks.shape}")

#         WDL = dice_loss()

#         print(WDL(main_out, masks))


# tester()



        # acc_results = get_pixel_accuracy(preds, masks)
        # acc_results["epoch"]=0
        # acc_results["step"]="validation"

        # recall_results = get_pixel_recall(preds, masks)
        # recall_results["epoch"]=0
        # recall_results["step"]="validation"

        # IoU_results = get_IoU(preds, masks)
        # IoU_results["epoch"]=0
        # IoU_results["step"]="validation"

        # data = [acc_results, recall_results, IoU_results]

        # df=pd.DataFrame(data)
        # df.set_index("assessment", inplace=True)

        # print(df)

        # print(f"Model weight dtype: {next(model.parameters()).dtype}")

        # Testing Loss functions

        # WCE_loss = abundance_weighted_CE_loss()
        # print(WCE_loss(main_out, masks))

        # DS_loss = dice_loss(relative_abundance=None, average='macro')
        # print(DS_loss(main_out, masks))

        # lg_DS_loss = log_cosh_dice_loss()
        # print(lg_DS_loss(main_out, masks))
