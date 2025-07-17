import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import UnFold

class SparseCRF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.unfold = UnFold(9, dilation=1, padding=0, stride=1)
        self.c = 4

    def forward(self, pred, feat):
        '''
            pred: B, 1, H, W
            feat: B, C, H, W
        '''
        assert pred.shape[2::]==feat.shape[2::], "scale !eq"
        assert len(pred) == len(feat), "batch size !eq"
        sigma2 = 0.15 ** 2
        weight = 10.0
        
        pred_ws = self.unfold(pred)  ## b,1,w_s,h,w
        feat_ws = self.unfold(feat)  ## b,C,w_s,h,w
        
        ## b,w_s,h,w
        part1 = torch.exp(-torch.pow(feat_ws[:,:,self.c:self.c+1,:,:] - feat_ws, 2.0).sum(dim=1) / (2 * sigma2)) 
        ## b,w_s,h,w
        part2 = torch.abs(pred_ws[:,:,self.c:self.c+1,:,:] - pred_ws).squeeze(1)
        loss = (part1 * part2).mean() * weight
        return loss