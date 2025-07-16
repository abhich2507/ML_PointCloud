#module for DeepSets implementation
import torch#, torchvision, torchmetrics
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import QuantileTransformer

class ExU(nn.Module):
    def __init__(self, alpha=1.0):
        super(ExU, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
    
class DeepSetLayer(nn.Module):
    def __init__(self, in_features:int, out_features:int,  normalization:str = 'batchnorm', pool:str = 'rms') -> None :
        """
        DeepSets single layer
        :param in_features: input's number of features
        :param out_features: output's number of features
        :param attention: Whether to use attention
        :param normalization: normalization method - 'fro' or 'batchnorm'
        
        """
        super(DeepSetLayer, self).__init__()

        self.Gamma = nn.Linear(in_features, out_features)
        self.Lambda = nn.Linear(in_features, out_features)

        self.normalization = normalization
        self.pool = pool
        print("pool: ", pool)
        
        if normalization == 'batchnorm':
            self.bn = nn.BatchNorm1d(out_features, affine=False)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
       
        print("Set:", x, "shape: ", x.shape)
        if(self.pool == 'mean') : 
            x = self.Gamma(x) + self.Lambda(x - x.mean(dim=1, keepdim=True)) # -- the average is over the points -- #
        elif(self.pool == 'mean2') : 
            x = self.Gamma(x) + self.Lambda(x - x.mean(dim=2, keepdim=True))
        elif(self.pool == 'max') :
          
            max_vals, _ = x.max(dim=1, keepdim=True)
            x = self.Gamma(x) + self.Lambda(x - max_vals) # -- the max is over the points -- #
        elif(self.pool == 'sum') :
            x = self.Gamma(x) + self.Lambda(x - x.sum(dim=1, keepdim=True))
        elif(self.pool == 'std') :
            x = self.Gamma(x) + self.Lambda(x - x.std(dim=1, keepdim=True))
        elif(self.pool == 'min') :
            min_vals, _ = x.min(dim=1, keepdim=True)
            x = self.Gamma(x) + self.Lambda(x - min_vals)
        elif(self.pool == 'median') :
            x = self.Gamma(x) + self.Lambda(x - x.median(dim=1, keepdim=True).values)
        elif(self.pool == 'l2') :
            x = self.Gamma(x) + self.Lambda(x - x.norm(dim=1, keepdim=True))
        elif(self.pool == 'l1') :
            x = self.Gamma(x) + self.Lambda(x - x.norm(dim=1, p=1, keepdim=True))
        elif(self.pool == 'rms') :
            x = self.Gamma(x) + self.Lambda(x - torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)))
        

        return x




def init_weights_he(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class DeepSet(nn.Module):
    def __init__(self, in_features:int, feats:list, n_class:int, normalization:str = 'batchnorm', pool:str = 'rms') -> None:
        super(DeepSet, self).__init__()

        layers = []

        layers.append(DeepSetLayer(in_features=in_features, out_features=feats[0], normalization=normalization, pool=pool))
        for i in range(1, len(feats)):
            layers.append(nn.LeakyReLU())  # Use LeakyReLU for better gradient flow
            layers.append(DeepSetLayer(in_features=feats[i-1], out_features=feats[i], normalization=normalization, pool=pool))

        layers.append(DeepSetLayer(in_features=feats[-1], out_features=n_class, normalization=normalization, pool=pool))
        self.sequential = nn.ModuleList(layers)

        # **Apply He Initialization to all layers**
        self.apply(init_weights_he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.sequential:
            x = layer(x)

        x = x.mean(dim=1)  # Average over points
        out = x  # Remove log_softmax if not needed
        return out




