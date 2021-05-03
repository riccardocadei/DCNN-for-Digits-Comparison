import torch

from models import *
from training import *

run_experiment(MLP(n_classes=2), 
                use_auxiliary_loss=False, 
                aux_loss_weight=0.1,
                nb_epochs=30,
                weight_decay=1e-1,
                model_name="ConvNet",
                augment=False,
                batch_size=50,
                lr=1e-3,
                percentage_val=0.1,
                verbose=2, 
                plot=False)


    


