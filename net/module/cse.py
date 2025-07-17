#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
# coding=utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import weight_init, LayerNorm2D, PositionwiseFeedForward

class CSE(nn.Module):
    def __init__(self, d_in, d_hidden, n_lib=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_in, d_hidden, 1), nn.BatchNorm2d(d_hidden), nn.ReLU(),
            nn.Conv2d(d_hidden, d_hidden, 1), LayerNorm2D(d_hidden), nn.ReLU(),
        )
        self.multi_head = nn.MultiheadAttention(d_hidden, 8, batch_first=True)
        self.g = PositionwiseFeedForward(d_hidden, d_hidden+d_hidden)
        self.libs = nn.Embedding(n_lib, d_hidden)
        weight_init(self)

    def forward(self, x, tau=0.1):
        kv = self.conv(x).flatten(2).transpose(-1, -2)  ## B, L, C
        
        # if os.environ["CSE_Q"]=="softmax":
        #     q = (torch.softmax(kv, dim=1) * kv).sum(dim=1, keepdim=True)  ## B, 1, C
        # elif os.environ["CSE_Q"]=="mean":
        #     q = kv.mean(dim=1, keepdim=True)  ## B, 1, C
        # elif os.environ["CSE_Q"]=="center":
        #     c = int(kv.shape[1] // 2)
        #     q = kv[:, c:c+1, :]  ## B, 1, C
        # elif os.environ["CSE_Q"]=="maxpool":
        #     q = nn.MaxPool1d(kv.shape[1])(kv.transpose(-1, -2)).transpose(-1, -2)  ## B, 1, C
        # else:
        #     raise
        ## softmax
        q = (torch.softmax(kv, dim=1) * kv).sum(dim=1, keepdim=True)  ## B, 1, C
            
        n = len(q) // 2  ## B=2n
        q, attn = self.multi_head(q, kv, kv)  ## B, 1, C; B, 1, L
        q = F.normalize(self.g(q), p=2, dim=-1)  ## B, 1, C
        
        sim = torch.matmul(q.permute(1, 0, 2), q.permute(1, 2, 0).detach())
        assert len(sim)==1, f"{sim.shape} should be (1,B,B)"
        sim = sim[0] - 1e9 * torch.eye(2*n, device=sim.device)
        sim = torch.softmax(sim/tau, dim=-1)  ## 2n, 2n
        
        loss = {}
        
        libmap = torch.max(q.unsqueeze(1) @ F.normalize(self.libs.weight, p=2, dim=-1).T, dim=-1)[0]
        loss["lib_loss"] = 1.0 - libmap.mean()
        
        prob1 = torch.diagonal(sim, offset=n, dim1=0, dim2=1)[0:n]
        prob2 = torch.diagonal(sim, offset=-n, dim1=0, dim2=1)[0:n]
        p1 = -torch.log(prob1 + 1e-6).mean()
        p2 = -torch.log(prob2 + 1e-6).mean()
        loss["contrastive"] = (p1 + p2) / 2.0
        
        # margin_loss = -torch.log(attn.max(dim=-1)[0] - attn.min(dim=-1)[0]+1e-6).mean()
        # loss["margin_loss"] = margin_loss * 0.01
        
        l1 = nn.L1Loss()(attn[0:n], attn[n::])
        loss["attn_L1_loss"] = l1
        
        attn = attn.unflatten(-1, x.shape[2::])
        
        return attn, attn, loss