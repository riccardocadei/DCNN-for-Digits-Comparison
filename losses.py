import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import math
from datetime import datetime
from torch.nn.modules.loss import _Loss
from torch import Tensor


class AuxiliaryLoss(_Loss):
    def __init__(self, weight_classification, weight_inequality, reduction='mean'):
        super().__init__(reduction=reduction)
        self.reduction = reduction
        tot = weight_classification * 2 + weight_inequality
        if  tot != 1.0:
            raise ValueError("2 * weight_classification + weight_inequality must be equal to 1! But you gave: ", tot)
        self.weight_classification = weight_classification
        self.weight_inequality = weight_inequality


    def forward(self, digits: Tensor, ineq: Tensor,  target: Tensor):
        loss_class1 = F.cross_entropy(digits[:, :10], target[:, 1])
        loss_class2 = F.cross_entropy(digits[:, 10:], target[:, 2])
        loss_ineq = F.cross_entropy(ineq, target[:,0])
        loss = self.weight_classification * (loss_class1 + loss_class2) + self.weight_inequality * loss_ineq

        return loss.mean()
