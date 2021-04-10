import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import math

import dlc_practical_prologue as prologue
from other import *


# returns a split in train and validation data
def random_split(train_input, train_target, train_classes, percentage_valid=0.1):
    # shuffle data
    idx = torch.randperm(train_input.size(0))
    train_input = train_input[idx]
    train_target = train_target[idx]
    train_classes = train_classes[idx]
    # split 
    train_size = math.floor(train_input.size(0) * (1 - percentage_valid))
    valid_size = train_input.size(0) - train_size
    train_input, valid_input = torch.split(train_input, [train_size, valid_size])
    train_target, valid_target = torch.split(train_target, [train_size, valid_size])
    train_classes, valid_classes = torch.split(train_classes, [train_size, valid_size])
    return train_input, train_target, train_classes, valid_input, valid_target, valid_classes

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


def run_experiment(model, nb_epochs = 25, weight_decay = 0.1, 
                            mini_batch_size = 50, lr = 1e-3*0.5, percentage_valid=0.1, verbose=False):

    # device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device used: ", device)

    # loading the data
    N = 1000 
    (train_input, train_target, train_classes,
     test_input, test_target, test_classes) = prologue.generate_pair_sets(N)
    print("Loading training and test set...")

    # splitting the dataset and data augmentation
    (train_input, train_target, train_classes,
     valid_input, valid_target, valid_classes) = random_split(train_input, train_target, train_classes, percentage_valid)
    print("Splitting the training set in training and validation set...")
    train_input, train_target, train_classes = augment(train_input, train_target, train_classes)
    print("Data augmentation...")
    print("In total there are: \n - {} samples in the Training Set ({} *2), \n - {} samples in the Validation Set, \n - {} samples in the Test Set"
        .format(train_input.size(0), int((1-percentage_valid)*N), int(percentage_valid*N), N))

    print('Number of parameters of the model: {}'.format(count_parameters(model)))

    # move to Device
    model.to(device)
    train_input.to(device)
    train_target.to(device)
    valid_input.to(device)
    valid_target.to(device)
    test_input.to(device)
    test_target.to(device)
    
    # training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    print('Training...')
    train_losses, valid_losses = train(model, train_input, train_target, valid_input, valid_target, optimizer, criterion, 
                                        nb_epochs=nb_epochs,  mini_batch_size=mini_batch_size, verbose=verbose)

    # evaluate the performances
    test_error = test(model, test_input, test_target)
    print('\nTest error: {0:.3f} %'.format(test_error*100) )

    return train_losses, valid_losses, test_error



def train(model, train_input, train_target, valid_input, valid_target, 
                 optimizer, criterion, nb_epochs = 25, mini_batch_size=50, verbose=True):
    """
    Train model
    """
    train_losses = []
    valid_losses = []
    for epoch in range(nb_epochs):
        train_loss_e = 0
        num_batches = 0
        # train batch
        for b in range(0, train_input.size(0), mini_batch_size):
            # Forward step
            output = model(train_input.narrow(0, b, mini_batch_size))
            # Compute the Loss
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

            # Backward step
            model.zero_grad()
            loss.backward()
            # Update the Gradient
            optimizer.step()
            # Collect the Losses
            train_loss_e += loss.data.item()
            num_batches += 1
        train_loss = train_loss_e / num_batches
        train_losses.append(train_loss)

        # validation
        valid_preds = model(valid_input)
        valid_loss = criterion(valid_preds, valid_target).data.item()
        valid_losses.append(valid_loss)

        if verbose:
            print("Epoch", epoch+1, "/", nb_epochs, "train loss:", train_loss, "valid loss:", valid_loss)

    return train_losses, valid_losses


def test(model, test_input, test_target):
    predict_classes_perc = model(test_input)
    _, predicted_classes = predict_classes_perc.max(1)
    test_error = (predicted_classes-test_target).abs().sum() / test_target.size(0)
    return test_error
    
