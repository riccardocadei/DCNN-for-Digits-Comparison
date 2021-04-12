import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import math
from datetime import datetime
from torch.nn.modules.loss import _Loss
from torch import Tensor


# initial dataset has images with two channels, each channel contains one
# greyscale image of a digit
# if augment = True we want to get a random combination of these digits every time
# we extract one item
class TrainDataset:
    def __init__(self, train_images: Tensor, train_target: Tensor, train_classes: Tensor, augment: bool,
                                         use_auxiliary_loss: bool):
        self.use_auxiliary_loss = use_auxiliary_loss
        self.augment = augment
        if not augment:
            self.images = train_images
            self.train_target = train_target
            self.train_classes = train_classes
        else:
            size_old = train_images.size(0)
            # store one channel images
            images = torch.empty(size_old * 2, 1, train_images.size(2), train_images.size(3))
            images[0:size_old] = train_images[:, 0].view(size_old, 1, train_images.size(2), train_images.size(3))
            images[size_old:2 * size_old] = train_images[:, 1].view(size_old, 1, train_images.size(2), train_images.size(3))
            self.images = images

            # store class of one channel images
            classes = torch.empty(size_old * 2)
            classes[0:size_old] = train_classes[:, 0]
            classes[size_old:2*size_old] = train_classes[:, 1]

            self.train_classes = classes
            self.train_target = None

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        if not self.augment:
            target = build_target(self.train_target[idx], self.train_classes[idx], self.use_auxiliary_loss)
            return self.images[idx], target
        
        else:
            img1 = self.images[idx]
            class1 = self.train_classes[idx]

            idx2 = torch.randint(low=0, high=len(self), size=(1,))
            img2 = self.images[idx2]
            class2 = self.train_classes[idx2]

            train_classes = torch.tensor([class1, class2])
            train_target = (class1 <= class2).int()

            image = torch.empty(2, img1.size(1), img1.size(2))
            image[0] = img1
            image[1] = img2

            target = build_target(train_target, train_classes, self.use_auxiliary_loss)
            return image, target



# returns a split in train and validation data
def random_split(train_input, train_target, train_classes, percentage_val=0.1):
    # shuffle data
    idx = torch.randperm(train_input.size(0))
    train_input = train_input[idx]
    train_target = train_target[idx]
    train_classes = train_classes[idx]
    # split 
    train_size = math.floor(train_input.size(0) * (1 - percentage_val))
    val_size = train_input.size(0) - train_size
    train_input, val_input = torch.split(train_input, [train_size, val_size])
    train_target, val_target = torch.split(train_target, [train_size, val_size])
    train_classes, val_classes = torch.split(train_classes, [train_size, val_size])
    return train_input, train_target, train_classes, val_input, val_target, val_classes

# since our task is to predict whether the first channel of images in train_input
# is lesser or equal than the second channel, we can flip the two channels and double our
# dataset size
def augment(train_input, train_target, train_classes):
    flipped_input = torch.empty(train_input.size())
    flipped_target = torch.empty(train_target.size())
    flipped_classes = torch.empty(train_classes.size())

    flipped_input[:,0] = train_input[:, 1].clone()
    flipped_input[:,1] = train_input[:, 0].clone()

    flipped_target = ((train_classes[:,1]-train_classes[:,0])<=0).int()

    flipped_classes[:,0] = train_classes[:,1].clone()
    flipped_classes[:,1] = train_classes[:,0].clone()
    augmented_input = torch.cat((train_input, flipped_input), dim=0)
    augmented_target = torch.cat((train_target, flipped_target), dim=0)
    augmented_classes = torch.cat((train_classes, flipped_classes), dim=0)
    return augmented_input, augmented_target, augmented_classes



def build_target(train_target, train_classes, use_auxiliary_loss):
    if not use_auxiliary_loss:
        return train_target
    else:
        target = torch.empty(train_target.size(0), 3)
        target[:, 0] = train_target
        target[:, 1:] = train_classes
        return target.long()


