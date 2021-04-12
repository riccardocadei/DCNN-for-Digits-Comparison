import torch

from dlc_practical_prologue import *

from models import *
from training import *
from other import *


model = IneqCNet(n_classes=22)

train_losses, test_losses, test_error = run_experiment(model, use_auxiliary_loss=True, nb_epochs=100, mini_batch_size=100, 
                                        weight_decay=1e-1 , lr=1e-4, verbose=2)

