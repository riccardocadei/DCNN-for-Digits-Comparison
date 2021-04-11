import torch

from dlc_practical_prologue import *

from models import *
from training import *
from other import *


model = ResNet(depth = 2, n_classes=2)

train_losses, test_losses, test_error = run_experiment(model, weight_decay=0, lr=1e-3, verbose=2)