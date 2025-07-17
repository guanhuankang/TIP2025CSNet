import torch
import torch.nn as nn
import torch.nn.functional as F


class BTM(nn.Module):
    def __init__(self):
        super().__init__()

    def get_contour(self, label):
        lbl = label.gt(0.5).float()
        ero = 1 - F.max_pool2d(1 - lbl, kernel_size=5, stride=1, padding=2)  # erosion
        dil = F.max_pool2d(lbl, kernel_size=5, stride=1, padding=2)            # dilation
        
        edge = dil - ero
        return edge

    # Boundary-aware Texture Matching Loss
    def BTMLoss(self, pred, image, radius=5):
            alpha = 200.0
            sal_map =  F.interpolate(pred, scale_factor=1.0, mode='bilinear', align_corners=True)
            image_ = F.interpolate(image, size=sal_map.shape[-2:], mode='bilinear', align_corners=True)
            mask = self.get_contour(sal_map)
            features = torch.cat([image_, sal_map], dim=1)
            
            N, C, H, W = features.shape
            diameter = 2 * radius + 1
            kernels = F.unfold(features, diameter, 1, radius).view(N, C, diameter, diameter, H, W)
            kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
            dis_sal = torch.abs(kernels[:, 3:])
            dis_map = (-alpha * kernels[:, 0:3] ** 2).sum(dim=1, keepdim=True).exp()
            distance = dis_map * dis_sal

            loss = distance.view(N, 1, (radius * 2 + 1) ** 2, H, W).sum(dim=2)
            loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-6)
            return loss

    def forward(self, pred, feat):
        return self.BTMLoss(pred=pred, image=feat, radius=5)