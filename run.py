import torch

from dlc_practical_prologue import *

from models import *
from training import *
from other import *


model = IneqCNet()

train_losses, test_losses, test_error = run_experiment(model, verbose=2)