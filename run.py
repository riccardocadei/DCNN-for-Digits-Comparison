import torch

from models import *
from training import *



# model
model = ResNetAux(depth=10,filters=64)
model_name='ResNet'

# training
percentage_val = 0.1
nb_epochs = 100
mini_batch_size = 100

# auxiliary loss
use_auxiliary_loss = use_aux_loss(model)
aux_loss_weight = 0.3   # <=0.5

# optimizer
weight_decay = 0.1
lr = 5e-4
period = 1

# display
verbose = 1
plot = True

run_experiment(model, use_auxiliary_loss=use_auxiliary_loss, aux_loss_weight=aux_loss_weight, model_name=model_name, percentage_val=percentage_val,nb_epochs=nb_epochs, mini_batch_size=mini_batch_size, weight_decay=weight_decay, lr=lr, period=period, verbose=verbose, plot=True);

