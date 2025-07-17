import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone
from . import neck
from . import module
from .loss import SparseCRF

from net.utils import weight_init, CRF, LocalWindowTripleLoss

def min2D(m):
    return torch.min(torch.min(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]

def max2D(m):
    return torch.max(torch.max(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]

def minMaxNorm(m, eps=1e-12):
    return (m - min2D(m)) / (max2D(m) - min2D(m) + eps)

def uphw(x, size):
    return F.interpolate(x, size=size, mode="bilinear")

class CSNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        logging.info(f"create {self.__class__}")
        self.backbone = getattr(backbone, cfg.BACKBONE)(init_weight=cfg.backboneWeight)
        self.decoder = getattr(neck, cfg.NECK)(dim_bin = [2048, 1024, 512, 256, 64])
        self.cse = getattr(module, cfg.MODULE)(2048, 512, cfg.cse_n_lib)

        self.head = nn.Sequential(
            nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        weight_init(self.head)
        
        self.sparseCRF= SparseCRF(cfg=cfg)
        self.crf = CRF()
        self.lwt = LocalWindowTripleLoss(cfg=cfg)
        self.cfg = cfg
        
    def forward(self, x, mask=None, process=0.0, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        pred = self.head(f1)

        if self.training:
            self.process = process
            size = pred.shape[2::]
            n = len(x) // 2
            division = float(self.cfg.division)
            
            attn_pos, attn_neg, loss = self.cse(f5, tau=self.cfg.cse_tau)
            if process <= division:
                if mask is None:
                    # pseudo = minMaxNorm(uphw(attn_pos, size=size)).gt(0.5).float()
                    pseudo = self.crf(uphw(minMaxNorm(x), size=size), minMaxNorm(uphw(attn_pos.detach(), size=size)), iters=self.cfg.crf_round).gt(0.5).float()
                else:
                    loss = {}  ## clear
                    pseudo = uphw(mask, size=size)
                bce_pr_wrt_ps = F.binary_cross_entropy_with_logits(pred, pseudo, reduction="none").reshape(n, -1).mean(dim=-1)
                loss["p1_bce"] = (bce_pr_wrt_ps * torch.softmax(self.cfg.rew_kai / (bce_pr_wrt_ps + 1.0), dim=-1)).sum()
                loss["p1_cons"] = F.l1_loss(torch.sigmoid(pred[0:n]), torch.sigmoid(pred[n::]))
            if process > division:
                loss["p2_lwt"] = self.lwt(torch.sigmoid(pred), minMaxNorm(x), margin=0.5)
                loss["p2_cons"] = F.l1_loss(torch.sigmoid(pred[0:n]), torch.sigmoid(pred[n::]))
                loss["p2_reg"] = 0.5 - torch.abs(torch.sigmoid(pred) - 0.5).mean()
                # loss["p2_scrf"] = self.sparseCRF(torch.sigmoid(pred), uphw(minMaxNorm(x), size=size))
            
            ## loss ablation
            for k in loss:
                if k not in self.cfg.loss_components:
                    loss[k] = loss[k] * 0.0
            return loss
        else:
            pred = uphw(torch.sigmoid(pred), size=x.shape[2::])
            
        return {
            "pred": pred
        }