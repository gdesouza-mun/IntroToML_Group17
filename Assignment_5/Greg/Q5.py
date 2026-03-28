from assign5 import load_data, error_rate
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#I'm importing the load_data from the assign5.py file



def display_random_grid(images, rows=10, cols=10, random_seed=None, invert=False, vmax=1.0):
    """Display a rows x cols grid of random images."""
    if random_seed is not None:
        np.random.seed(random_seed)

    n_needed = rows * cols
    indices = np.random.choice(len(images), n_needed, replace=False)

    # Create grid
    grid = np.zeros((rows * 28, cols * 28))
    for i, idx in enumerate(indices):
        r, c = i // cols, i % cols
        grid[r*28:(r+1)*28, c*28:(c+1)*28] = images[idx]

    plt.figure(figsize=(12, 12))
    #Vmax determine the maximum value of the image (colored black)
    #If before tensor normalization, vmax=255
    #For Tensors let vmax=1
    plt.imshow(grid, cmap='gray_r', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.show()

def img_augmenter(img, chance=0.5):
    '''
    Given an image, it has a chance of applying a transformation to the function
    '''

    out_img = img
    if random.random() < chance:
        choice = random.randint(0,3)

        #50% - Applies a random x and y translation
        if choice == 0 or choice == 3:
            row_shift = random.randint(-5,5)
            col_shift = random.randint(-5,5)
            shifter = transform.AffineTransform(translation=(row_shift, col_shift))
            out_img = transform.warp(img,  shifter.inverse, preserve_range=True).astype(img.dtype)


        #50% applies a random rotation
        if choice == 1:
            multi = random.randint(1,3)
            out_img =  transform.rotate(img, angle=multi*5,
                                        preserve_range=True,
                                        cval=0, mode='constant',
                                        resize=False)
        if choice == 2:
            multi = random.randint(1,3)
            out_img = transform.rotate(img, angle=multi*(-5),
                                       preserve_range=True,
                                       cval=0, mode='constant',
                                       resize=False)

    return out_img



class AugmentedMNIST(Dataset):
    '''
    This is a torch dataset, used to handle getting data items
    from a collection of items, here I'm passing everything
    as arrays like (N, 28,28), but it can also be directories
    or other more elaborate structures
    '''
    def __init__(self, data, targets, augment=False):
        #Init if data and targets
        self.data = data
        self.targets = targets

        #If augment is true, I'll call the augmentation function
        #When getting the images
        self.augment=augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #This is the crucial method, it allows images to be called
        #by the [ ] method, like an array

        #So given an index, it gets the image and the target associeted with
        #that index
        img = self.data[idx]
        label = self.targets[idx]

        #if augment, I apply the augment function
        if(self.augment):
            img = img_augmenter(img, 0.5)

        #torch expected tensors from 0 to 1, so I rescale the image
        img_tensor = torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0)
        #The unsqueeze method adds a channel dimension, so we go from
        #(28x28) to (1, 28x28), this channels will be used in the Neural Network
        label_tensor = torch.tensor(label, dtype=torch.long)

        #I also set a mean and std based on MNIST training reccomendations
        mean = 0.1307
        std = 0.3081

        img_tensor = (img_tensor - mean)/std

        return img_tensor, label_tensor



class SmallCNN(nn.Module):
    '''
    This is a small convolutional neural network, that will
    make predictions based on 28x28 images as input.
    '''
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        #First layer - Apply a 3x3 convolution going from 1 to 32 channels
        #So now, instead of 1,28x28 our data becomes 32x28,28

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #Batch norm just makes sure the parameters don't explode
        self.bn1 = nn.BatchNorm2d(32)

        #Same thing, but going from 32 to 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #Max pools(2,2) for each 4 pixel, it picks the max pixel, reducing the size
        #Now our image is 64 channels and 7x7
        self.pool = nn.MaxPool2d(2, 2)

        #Randomly sets 0.3 parameters to 0, so the network learns alternate paths
        self.dropout1 = nn.Dropout2d(0.3)

        #We pass through two linear layes to reduce from 64x7x7 to 10 labels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #We just call the network in order relu-ing where necessary
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_one_epoch(model, loader, optimizer, criterion, scheduler, device):
    #For one epoc

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        if data.dim()==3:
            data = data.unsqueeze(1)

        #This moves the data to the proper device if we are doing GPU training
        data, target = data.to(device), target.to(device)

        #Zero grad, generate output, evaluate loss
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)

        #Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        #The scheduler allows for dynamic learning rate
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return running_loss/len(loader), correct/total


def train_model(epochs=30, save=False, final=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, y_train, X_test, y_test = load_data("data")

    model = SmallCNN()
    model = model.to(device)

    if final:
        X = np.vstack((X_train, X_test))
    else:
        X=X_train

    train_dataset = AugmentedMNIST(X, y_train, True)
    test_dataset = AugmentedMNIST(X_test, y_test, False)

    batch_size=64

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False)

    criterion= nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                          steps_per_epoch=len(train_loader),
                                          epochs=epochs)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                optimizer, criterion,
                                                scheduler, device)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if data.dim() == 3: data=data.unsqueeze(1)
                outputs = model(data)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(target).sum().item()
        val_acc = val_correct/len(test_dataset)

        print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}%")

train_model(epochs=50, final=False)
