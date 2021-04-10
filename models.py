import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class IneqNET(nn.Module):
    """
    Description
    """
    def __init__(self):
        super(IneqNET, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, padding = 3)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=3, padding = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(720)
        self.fc1 = nn.Linear(720, 100)
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self, x):
        """
        General structure of one layer:
            Input -> Convolution -> BatchNorm -> Activation(ReLu) -> Maxpooling -> Output
        """
        # 1st layer 
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2)
        # 2nd layer 
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2)
        # 3rd layer
        x = F.relu(self.fc1(self.bn3(x.view(x.size()[0], -1))))
        # 4th layer
        x = self.fc2(F.dropout(x)) 
        
        return x

class IneqMLP(nn.Module):
    """
    MLP with 4 layers
    """
    def __init__(self):
        super(IneqMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(392, 196),
            nn.ReLU(),
            nn.Linear(196, 98),
            nn.ReLU(),
            nn.Linear(98, 49),
            nn.ReLU(),
            nn.Linear(49, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        
    def forward(self, x):
        """
        General structure of one layer:
            Input -> Linear -> Activation(ReLu) -> Output
        """
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x