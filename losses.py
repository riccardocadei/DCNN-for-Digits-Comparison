import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import math
from datetime import datetime
from torch.nn.modules.loss import _Loss
from torch import Tensor


class AuxiliaryLoss(_Loss):
    def __init__(self, reduction='mean', weight_classification=0.2 , weight_inequality=0.6):
        super().__init__(reduction=reduction)
        self.reduction = reduction
        tot = weight_classification * 2 + weight_inequality
        if  tot != 1.0:
            raise ValueError("2 * weight classification + weight of inequality must be 1!But you gave:", tot)
        self.weight_classification = weight_classification
        self.weight_inequality = weight_inequality

    # preds is of size: (N, 2 (inequality) + 10 (class1) + 10 (class2))
    # target is of size: (N, 1 (inequality) + 1 (class1) + 1 (class2))
    def forward(self, preds: Tensor, target: Tensor):
        loss_ineq = F.cross_entropy(preds[:, :2], target[:,0])
        loss_class1 = F.cross_entropy(preds[:, 2:12], target[:, 1])
        loss_class2 = F.cross_entropy(preds[:, 12:22], target[:, 2])
        loss = self.weight_classification * (loss_class1 + loss_class2) + self.weight_inequality * loss_ineq
        return loss.mean()