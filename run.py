import torch

from models import *
from training import *

model_params = [2]


evaluate_model(MLP, 
                *model_params,
                use_auxiliary_loss=False, 
                aux_loss_weight=0.1,
                nb_epochs=25,
                weight_decay=1e-4,
                model_name="ConvNet",
                augment=False,
                batch_size=50,
                lr=1e-3,
                percentage_val=0.1,
                verbose=1
                )