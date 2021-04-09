import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import dlc_practical_prologue as prologue

def train_IneqNET(model, train_input, train_target, nb_epochs = 25, \
                 lambda_l2 = 0.1, mini_batch_size = 50, lr = 1e-3*0.5): 
    """
    Train IneqNET model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    
    train_losses = []
    for e in range(nb_epochs):
        train_loss_e = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            # Forward step
            output = model(train_input.narrow(0, b, mini_batch_size))
            # Compute the Loss
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            for p in model.parameters():
                loss += lambda_l2 * p.pow(2).sum()
            # Backward step
            model.zero_grad()
            loss.backward()
            # Update the Gradient
            optimizer.step()
            # Collect the Losses
            train_loss_e += loss.data.item()
        train_losses.append(train_loss_e/mini_batch_size)
    return train_losses