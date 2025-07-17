import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import UnFold

class LocalAppearanceTriplet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.unfold = UnFold(11, dilation=1, padding=0, stride=5)
        self.kai = cfg.kai
        self.margin = cfg.margin

    def forward(self, pred, feat):
        '''
            pred: B, 1, H, W
            feat: B, C, H, W
        '''
        assert pred.shape[2::]==feat.shape[2::], "scale !eq"
        assert len(pred) == len(feat), "batch size !eq"
        eps = 1e-6
        kai = self.kai
        margin = self.margin
        
        pred_ws = self.unfold(pred)  ## b,1,w_s,h,w
        feat_ws = self.unfold(feat)  ## b,C,w_s,h,w
        
        pos = (pred_ws * feat_ws).sum(dim=2, keepdim=True) / (pred_ws.sum(dim=2, keepdim=True) + eps)
        neg = ((1.0-pred_ws) * feat_ws).sum(dim=2, keepdim=True) / ((1.0-pred_ws).sum(dim=2, keepdim=True) + eps)
        
        dpos = 1.0 - torch.exp( -kai * torch.sum(torch.pow(feat_ws-pos, 2.0), dim=1, keepdim=True) )
        dneg = 1.0 - torch.exp( -kai * torch.sum(torch.pow(feat_ws-neg, 2.0), dim=1, keepdim=True) )
        dist = (pred_ws - 0.5) * (dpos - dneg)
        loss_lwt = torch.maximum(dist + margin, torch.zeros_like(dist))
        return loss_lwt.mean()
        
        # valid = ((torch.sum(m, dim=2, keepdim=True) > 0.5) * (torch.sum(1.-m, dim=2, keepdim=True) > 0.5)).float() ## no gradient
        # loss = torch.sum(triple * valid.detach(), dim=[3,4]) / (torch.sum(valid.detach(), dim=[3,4]) + 1e-6)