import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import math

class IneqCNet(nn.Module):
    """
    Description
    """
    def __init__(self, use_auxiliary_loss):
        super(IneqCNet, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, padding = 3)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=3, padding = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(720)
        self.fc1 = nn.Linear(720, 100)
        self.fc2 = nn.Linear(100, n_classes)
        
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

class IneqCNetAux(nn.Module):
    """
    Description
    """
    def __init__(self):
        super(IneqCNetAux, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, padding = 3)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=3, padding = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(720)
        self.fc1 = nn.Linear(720, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 2)
        
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
        # 4rd layer
        x_class = F.relu(self.fc2(x))
        # 5th layer
        x = self.fc3(x_class) 
        
        return x_class, x

class IneqMLP(nn.Module):
    """
    MLP with 4 layers
    """
    def __init__(self, n_classes=2):
        super(IneqMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(392, 350),
            nn.ReLU(),
            nn.Linear(350, 250),
            nn.ReLU(),
            nn.Linear(250, 200),
            nn.ReLU(),
            nn.Linear(200, 20),
            nn.ReLU(),
            nn.Linear(20, n_classes),
        )
        
    def forward(self, x):
        """
        General structure of one layer:
            Input -> Linear -> Activation(ReLu) -> Output
        """
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x




class ResidualBlock(nn.Module):
    def __init__(self, filters, input_channels, conv_shortcut = False,  kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.filters = filters
        self.input_channels = input_channels
        self.conv_shortcut = conv_shortcut
        if self.conv_shortcut:
            self.conv_sc = nn.Conv2d(in_channels=input_channels, out_channels= 4*filters, kernel_size=1, stride=stride)
            self.bn_sc = nn.BatchNorm2d(num_features=4*filters)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=filters)
    
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=filters)
        
        #conv3 keeps image size
        self.conv3 = nn.Conv2d(in_channels=filters, out_channels=4 * filters, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_features=4*filters)
        
    def forward(self, x):
        if self.conv_shortcut:
            shortcut = self.bn_sc(self.conv_sc(x))
        else:
            shortcut = x
       # print("shortcut size:", shortcut.size())
        x = F.relu(self.bn1(self.conv1(x)))
       #print("after first convolution", x.size())
        padding = math.ceil(0.5 * (x.size()[2] * (self.stride - 1) + self.kernel_size - self.stride))
        #print(padding)
        pad = nn.ZeroPad2d(padding)
        x = pad(x)
        #print("after padding", x.size())
        x = F.relu(self.bn2(self.conv2(x)))
      #  print("after second convolution", x.size())
        x = self.bn3(self.conv3(x))
       # print("after third convolution", x.size())
        x = torch.add(x, shortcut)
        x = F.relu(x)
        return x
      


class ResNetAux(nn.Module):
    def __init__(self, depth, n_classes=22, input_channels=2, filters=32, input_size=14):
        super(ResNetAux, self).__init__()
        self.depth = depth
        self.input_channels = input_channels
        # residual blocks keep the channels with same size as input images
        blocks = []
        blocks.append(ResidualBlock(filters=filters, input_channels=2, conv_shortcut=True))
        for i in range(depth - 1):
            blocks.append(ResidualBlock(filters=filters, input_channels=4*filters))

        self.blocks = nn.ModuleList(blocks)
        self.avg_pool = nn.AvgPool2d(kernel_size=input_size) #global average pooling
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(in_features=4*filters, out_features=n_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        return x[:,:20], x[:,20:]




################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def use_aux_loss(model):
    if type(model).__name__[-3:]=='Aux': 
        return True
    else:
        return False
