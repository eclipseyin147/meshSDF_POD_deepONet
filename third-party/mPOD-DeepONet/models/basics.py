import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PermuteLayer(nn.Module):
    def __init__(self, dims:tuple):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return torch.permute(x, self.dims)