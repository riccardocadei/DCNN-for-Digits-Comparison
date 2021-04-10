import torch

from dlc_practical_prologue import *

from models import *
from training import *
from other import *


model = IneqNET()

train_losses, test_losses, test_error = run_experiment_no_aux(model)