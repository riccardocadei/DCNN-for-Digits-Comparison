import torch

from models import *
from training import *


models = [MLP, ConvNet, ResNet, DeepConvNet, DeepConvNet, Siamese, Siamese] 
model_names = ["MLP", "ConvNet", "ResNet", "DeepConvNet", "DeepConvNet", "Siamese", "Siamese"] 

model_params = [[2], # MLP 2 classes
                [2], # ConvNet 2 classes
                [15, 2, 2, 128], # Resnet depth 15, 2 classes, 2 input channels, 128 channels in ConvBlocks
                [False], #DeepConvNet without aux loss
                [True], #DeepConvNet with aux loss
                [False], #Siamese without aux loss
                [True]] #Siamese with aux loss


weight_decays = {"MLP": 1e-1,
                 "ConvNet": 1e-1,
                 "ResNet": 1e-4,
                 "DeepConvNet" : 1e-5,
                 "Siamese": 1e-5}

learning_rates = {"MLP": 1e-4,
                 "ConvNet": 1e-4,
                 "ResNet": 1e-4,
                 "DeepConvNet" : 1e-4,
                 "Siamese": 1e-4}

use_augment = [False, True] 
epochs = [25, 200] # 25 epochs without augmentation, 200 epochs with augmentation
num_experiments = [10, 3] # 10 experiments without augmentation, 3 with augmentation


mean_train_errors = []
std_train_errors = []
mean_val_errors = []
std_val_errors = []
mean_test_errors = []
std_test_errors = []


for augment, nb_epochs, n_experiment in zip(use_augment, epochs, num_experiments):
    for model, model_name, params in zip(models, model_names, model_params):
        lr = learning_rates[model_name]
        weight_decay = weight_decays[model_name]
        use_aux_loss = (model_name == "Siamese" or model_name == "DeepConvNet") and params[0] == True

        ((mean_train_error,std_train_error), 
            (mean_val_error,std_val_error), 
            (mean_test_error,std_test_error)) = evaluate_model(model,
                                                             *params, 
                                                            n_experiments = n_experiment,
                                                            use_auxiliary_loss = use_aux_loss,
                                                            aux_loss_weight=0.3, 
                                                            model_name=model_name,
                                                            nb_epochs = nb_epochs, 
                                                            weight_decay = weight_decay,
                                                            augment=augment,
                                                            batch_size = 50,
                                                            lr = lr, 
                                                            percentage_val=0.1,
                                                            verbose=1)
        mean_train_errors.append(mean_train_error)
        std_train_errors.append(std_train_error)
        mean_val_errors.append(mean_val_error)
        std_val_errors.append(std_val_error)
        mean_test_errors.append(mean_test_error)
        std_test_errors.append(std_test_error)

            

