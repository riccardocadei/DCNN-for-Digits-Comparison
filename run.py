import torch

from dlc_practical_prologue import *

from models import *
from training import *
from other import *


model = IneqCNet(n_classes=22)

train_losses, test_losses, test_error = run_experiment(model, use_auxiliary_loss=True, mini_batch_size=100, 
                                        weight_decay=0 , lr=1e-3, verbose=2)