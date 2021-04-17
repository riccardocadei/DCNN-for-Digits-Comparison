import torch

from models import *
from training import *

run_experiment(ConvNet(depth=30, use_auxiliary_loss=True, n_classes=2, filters=32), 
                use_auxiliary_loss=True, 
                aux_loss_weight=0.3,
                nb_epochs=25,
                weight_decay=1e-4,
                model_name="ConvNet",
                augment=True,
                batch_size=50,
                lr=1e-3,
                percentage_val=0.1,
                verbose=2, 
                plot=False)