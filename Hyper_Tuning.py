from hyperopt import tpe, hp, fmin
import numpy as np
import torch
import dlc_practical_prologue as prologue
from models import *
from training import *
import optuna

def objective(trial): 
    aux_loss_weight=trial.suggest_float('aux_loss_weight', 0.1, 0.5)
    weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-1, log = True)
    lr=trial.suggest_float('lr', 1e-5, 1e-1, log = True)       
    filters=trial.suggest_int('filters', 8, 100)
    train_losses, val_losses, (train_error, val_error, test_error) = run_experiment(ConvNet(depth=30, use_auxiliary_loss=True, n_classes=2, filters=filters), 
                                                                                    use_auxiliary_loss=True, 
                                                                                    aux_loss_weight=aux_loss_weight,
                                                                                    nb_epochs=50,
                                                                                    weight_decay=weight_decay,
                                                                                    model_name="ConvNet",
                                                                                    augment=True,
                                                                                    batch_size=50,
                                                                                    lr=lr,
                                                                                    percentage_val=0.1,
                                                                                    verbose=0, 
                                                                                    plot=False)
    return test_error



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)