import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import math
from datetime import datetime
from torch.nn.modules.loss import _Loss
from torch import Tensor
import time

import dlc_practical_prologue as prologue
from plot import *
from models import count_parameters, use_aux_loss
from losses import AuxiliaryLoss
from data_helpers import random_split, build_target, augment, TrainDataset


def get_criterion(use_auxiliary_loss, weight_classification):
    if not use_auxiliary_loss:
        return nn.CrossEntropyLoss()
    else:
        return AuxiliaryLoss(weight_classification=weight_classification,
                                         weight_inequality=1-2*weight_classification)
        


def run_experiment(model, use_auxiliary_loss, aux_loss_weight=0.3, nb_epochs = 25, weight_decay = 0.1, model_name="model", period=1,
                            mini_batch_size = 50, lr = 1e-3*0.5, percentage_val=0.1, verbose=1, plot=True):

    # device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose>=1: print("Device used: ", device)

    # loading the data
    N = 1000 
    (train_input, train_target, train_classes,
     test_input, test_target, test_classes) = prologue.generate_pair_sets(N)
    if verbose>=1: print("Loading training and test set...")

    # splitting the dataset and data augmentation
    (train_input, train_target, train_classes, val_input, val_target, val_classes) = random_split(train_input, train_target, train_classes, percentage_val)
    if verbose>=1: print("Splitted the training set in training and validation set")

    train_ds = TrainDataset(train_input, train_target, train_classes, augment=True, 
                                                        use_auxiliary_loss=use_auxiliary_loss)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=mini_batch_size, shuffle=False)
    
    if verbose>=1: print('Number of parameters of the model: {}'.format(count_parameters(model)))
    train_target = build_target(train_target, train_classes, use_auxiliary_loss)
    val_target = build_target(val_target, val_classes, use_auxiliary_loss)

    # move to device static data, training data is augmented, so it will be copied on device later
    model = model.to(device)

    train_input = train_input.to(device)
    train_target = train_target.to(device)
    val_input = val_input.to(device)
    val_target = val_target.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    # training
    criterion = get_criterion(use_auxiliary_loss, aux_loss_weight)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    if verbose>=1: print('Training...')
    start = time.time()
    train_losses, val_losses = train(model, use_auxiliary_loss, train_loader, val_input, val_target, optimizer, criterion, device, model_name=model_name,
                                            nb_epochs=nb_epochs,  mini_batch_size=mini_batch_size, verbose=verbose)
    end = time.time()
    if verbose >= 1: print('Training time: {0:.3f} seconds'.format(end-start))
    path = "./model_weights/" + model_name + ".pth"
    if verbose >= 1: print("The model weights have been correctly saved in: ", path)

    # load weights of best model in validation
    model.load_state_dict(torch.load(path))

    
    if use_auxiliary_loss: 
        train_target = train_target[:,0]
        val_target = val_target[:,0]
    # evaluate the performances
    train_error = test(model, use_auxiliary_loss, train_input, train_target, device)
    if verbose>=1: print('\nTraining error: {0:.3f} %'.format(train_error*100) )
    val_error = test(model, use_auxiliary_loss, val_input, val_target, device)
    if verbose>=1: print('Validation error: {0:.3f} %'.format(val_error*100) )
    test_error = test(model, use_auxiliary_loss, test_input, test_target, device)
    if verbose>=1: print('Test error: {0:.3f} %'.format(test_error*100) )

    if plot==True: plot_train_val(train_losses, val_losses, period=period, model_name=model_name)

    return train_losses, val_losses, (train_error, val_error, test_error)



def train(model, use_auxiliary_loss, train_loader, val_input, val_target, 
                 optimizer, criterion, device, model_name="model", nb_epochs = 25, mini_batch_size=50, verbose=2):
    """
    Train model
    """
    train_losses = []
    val_losses = []
    for epoch in range(nb_epochs):
        train_loss_e = 0
        model.train()
        # train batch
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            # Forward step
            if use_auxiliary_loss: 
                targets = targets.view(targets.size(0), -1).to(device)
                train_pred_d, train_pred_i = model(inputs)
                loss = criterion(train_pred_d, train_pred_i, targets)
            else:
                targets = targets.view(targets.size(0)).long().to(device)
                output = model(inputs)
                loss = criterion(output, targets)
            # Backward step
            model.zero_grad()
            loss.backward()
            # Update the Gradient
            optimizer.step()
            # Collect the Losses
            train_loss_e += loss.data.item()
        train_loss = train_loss_e / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        # validation
        if use_auxiliary_loss:
            val_pred_d, val_pred_i = model(val_input)
            val_loss = criterion(val_pred_d, val_pred_i, val_target).data.item()
        else:
            val_preds = model(val_input)
            val_loss = criterion(val_preds, val_target).data.item()
        val_losses.append(val_loss)
        # save best model in validation
        if val_loss <= min(val_losses):
            torch.save(model.state_dict(), "./model_weights/" + model_name + ".pth")
        if verbose==2:
            print("Epoch", epoch+1, "/", nb_epochs, "train loss:", train_loss, "valid loss:", val_loss)

    return train_losses, val_losses


def evaluate_model(model, n=10, use_auxiliary_loss=False, aux_loss_weight=0.3, nb_epochs = 25, weight_decay = 0.1, 
                    mini_batch_size = 50, lr = 1e-3*0.5, percentage_val=0.1, verbose=0):
    train_errors = []
    val_errors = []
    test_errors = []
    print('Number of experiments: {}'.format(n))
    print('Computing...')
    for i in range(n):
        _, _, errors = run_experiment(model, use_auxiliary_loss=use_auxiliary_loss, aux_loss_weight=aux_loss_weight, nb_epochs=nb_epochs, percentage_val=percentage_val, mini_batch_size=mini_batch_size, weight_decay=weight_decay, lr=lr, verbose=verbose, plot=False);
        train_errors.append(errors[0])
        val_errors.append(errors[1])
        test_errors.append(errors[2])
    mean_train_error = torch.mean(torch.Tensor(train_errors))
    mean_val_error = torch.mean(torch.Tensor(val_errors))
    mean_test_error = torch.mean(torch.Tensor(test_errors))
    std_train_error = torch.std(torch.Tensor(train_errors))
    std_val_error = torch.std(torch.Tensor(val_errors))
    std_test_error = torch.std(torch.Tensor(test_errors))
    print('Training Set: \n- Mean: {0:.3f}\n- Standard Error: {0:.3f}'.format(mean_train_error,std_train_error) )
    print('Validation Set: \n- Mean: {0:.3f}\n- Standard Error: {0:.3f}'.format(mean_val_error,std_val_error) )
    print('Test Set: \n- Mean: {0:.3f}\n- Standard Error: {0:.3f}'.format(mean_test_error,std_test_error) )
    return

def test(model, use_auxiliary_loss, test_input, test_target, device):
    model.eval()
    preds = torch.empty(test_target.size(0), 2).to(device)
    # avoid memory overflow
    batch_size = 20
    for i in range(0, test_input.size(0), batch_size):
      inputs = test_input.narrow(0, i, batch_size)
      with torch.no_grad():
         # select only the first two columns in case auxiliary loss is used in training
         if use_auxiliary_loss: _, outputs = model(inputs)
         else: outputs = model(inputs)
      preds[i : i + batch_size, :] = outputs
        
    _, predicted_classes = preds.max(1)
    test_error = (predicted_classes-test_target).abs().sum() / test_target.size(0)
    return test_error