import math
import torch
import torch.nn as nn

class DINOHub(nn.Module):
    def __init__(self, name):
        super().__init__()
        ## DINO v1: https://github.com/facebookresearch/dino
        ## DINO v2: https://github.com/facebookresearch/dinov2
        # dino_models = {
        #     "dinov2_vits14_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc'),
        #     "dinov2_vitb14_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc'),
        #     "dinov2_vitl14_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc'),
        #     "dinov2_vitg14_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc'),
        #     "dinov2_vits14_reg_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc'),
        #     "dinov2_vitb14_reg_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc'),
        #     "dinov2_vitl14_reg_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc'),
        #     "dinov2_vitg14_reg_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc'),
        #     "vits16": torch.hub.load('facebookresearch/dino:main', 'dino_vits16'),
        #     "vits8": torch.hub.load('facebookresearch/dino:main', 'dino_vits8'),
        #     "vitb16": torch.hub.load('facebookresearch/dino:main', 'dino_vitb16'),
        #     "vitb8": torch.hub.load('facebookresearch/dino:main', 'dino_vitb8'),
        #     "resnet50": torch.hub.load('facebookresearch/dino:main', 'dino_resnet50'),
        # }
        if name.startswith("dinov2"):
            self.model = torch.hub.load('facebookresearch/dinov2', name)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', name)
        

class DINO(DINOHub):
    def __init__(self, init_weight):
        super().__init__(init_weight)
        self.model_name=init_weight
        
    def adaptiveReshape(self, x):
        b, n, c = x.shape
        h = int(math.sqrt(n))
        w = n//h
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)
    
    def forward(self, x):
        out = {}
        m = self.model
        if self.model_name=="dino_resnet50":
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            out["res2"] = m.layer1(x)
            out["res3"] = m.layer2(out["res2"])
            out["res4"] = m.layer3(out["res3"])
            out["res5"] = m.layer4(out["res4"])
        elif self.model_name.startswith("dino_"):
            out["res2"] = self.adaptiveReshape(m.patch_embed(x))
            out["res3"] = self.adaptiveReshape(m.get_intermediate_layers(x)[0][:, 0:-1, :])  ## omit the cls token
        elif self.model_name.startswith("dinov2_"):
            out["res2"] = self.adaptiveReshape(m.backbone.patch_embed(x))
            out["res3"] = self.adaptiveReshape(m.backbone.get_intermediate_layers(x)[0])
        return out
    