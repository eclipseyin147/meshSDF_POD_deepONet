import torch
import torch.nn as nn

"""
Multivariate DeepONet Implementation

The architecture configuration for the Branch and Trunk networks in this script is heavily inspired by @xie-lab-ml DeepONet.
Source: https://github.com/xie-lab-ml/multiphysics-bench
"""

class MultiChannelBranchNet(nn.Module):
    def __init__(self, num_channels=3, p=256):
        super().__init__()
        self.p = p
        self.C = num_channels
        
        # This is exactly your structure
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # The key change: The output is (C * p)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.C * self.p) 
        )

    def forward(self, T0):
        x = self.conv_layers(T0)
        x = x.view(x.size(0), -1)
        # Reshape to [Batch, Channels, p]
        return self.fc(x).view(-1, self.C, self.p)
    

class ShareTrunkDeepONet(nn.Module):
    def __init__(self, p=256, out_channels=3, device='cuda'):
        super().__init__()
        self.C = out_channels
        self.p = p
        self.branch = MultiChannelBranchNet(out_channels, p)
        self.trunk = nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, p*out_channels)
            ) #TrunkNet(p)
        self.output_net = nn.Linear(p, out_channels)

        x = torch.linspace(-0.635, 0.635, 128)
        y = torch.linspace(-0.635, 0.635, 128)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        # [128*128, 2]
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        self.register_buffer('coords', coords)
        

    def forward(self, T0):
        # T0: [batch, 1, 128, 128]
        B, _, W, H = T0.shape
        b = self.branch(T0)  # [batch, p]
        t = self.trunk(self.coords).view(-1, self.C, self.p)  # [batch, num_points, p]
        output = torch.einsum('bcp, ncp -> bcn', b, t)
        return output.reshape(B, self.C, W, H) # [batch, 3]
    

class MultiTrunkDeepONet(nn.Module):
    def __init__(self, out_channels=3, p=256):
        super().__init__()
        self.C = out_channels
        self.p = p
        
        # Your Branch
        self.branch = MultiChannelBranchNet(out_channels, p)
        
        # List of Trunks (one per output channel)
        self.trunks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, p)
            ) for _ in range(out_channels)
        ])
        
        # Setup the 128x128 coordinate grid as you had it
        x = torch.linspace(-0.635, 0.635, 128)
        y = torch.linspace(-0.635, 0.635, 128)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        # [128*128, 2]
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        self.register_buffer('coords', coords)

    def forward(self, T0):
        B, _, W, H = T0.shape
        
        # 1. Branch output: [B, C, p]
        b_out = self.branch(T0)
        
        # 2. Trunk outputs: Collect each trunk's basis functions
        # Each trunk output: [16384, p] -> stacked to [C, 16384, p]
        t_outputs = torch.stack([trunk(self.coords) for trunk in self.trunks], dim=0)
        
        # 3. Fusion: Dot product over 'p' for each channel
        # b: batch, c: channel, p: basis, n: points (16384)
        out = torch.einsum('bcp, cnp -> bcn', b_out, t_outputs)
        
        # Reshape back to grid: [B, C, 128, 128]
        return out.reshape(B, self.C, W, H)
