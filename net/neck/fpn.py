import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import UnFold, weight_init

class FPNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(d_in, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(d_out, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        weight_init(self)

    def forward(self, x, ref):
        x = F.interpolate(self.conv1(x), size=ref.shape[2::], mode="bilinear")
        return x + self.conv2(ref)

class FPN(nn.Module):
    '''
        Cross-Scale Feature Re-Coordinate Pyramid Network
    '''
    def __init__(self, dim_bin = [2048, 1024, 512, 256, 64]):
        super().__init__()
        self.frc = nn.ModuleList([ FPNLayer(f_in, f_out) for f_in, f_out in zip(dim_bin[0:-2], dim_bin[1:-1]) ])
        self.conv1 = nn.Sequential(nn.Conv2d(dim_bin[-2], dim_bin[-1], 1), nn.BatchNorm2d(dim_bin[-1]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(dim_bin[-1], dim_bin[-1], 1), nn.BatchNorm2d(dim_bin[-1]), nn.ReLU())

    def forward(self, features):
        ''' features: f5,f4,f3,f2,f1 '''
        n = len(features)
        out = [features[0],]
        for i in range(n-2):
            out.append(self.frc[i](out[-1], features[i+1]))
        out.append( self.conv1(out[-1])+self.conv2(features[-1]) )
        return out