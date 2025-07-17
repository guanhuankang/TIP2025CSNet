# coding=utf-8
import logging
import torch
import torch.nn as nn

# from net import R50FrcPN
import net

class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.model = R50FrcPN(cfg)
        self.model = getattr(net, cfg.META_ARCHITECTURE)(cfg)

    def loadCheckPoint(self, init_weight):
        import os
        if isinstance(init_weight, str) and os.path.exists(init_weight):
            logging.info(f" network loads ckp from '{init_weight}' ")
            ckp = torch.load(init_weight)
            model_state_dict = self.state_dict()
            matched_keys = []
            unmatched_keys = []
            for k, v in ckp.items():
                if k in model_state_dict:
                    model_state_dict[k] = v
                    matched_keys.append(k)
                else:
                    unmatched_keys.append(k)
            self.load_state_dict(model_state_dict, strict=True)
            # logging.info("matched:" + str(matched_keys))
            logging.info("miss keys:" + str(unmatched_keys))
        else:
            logging.info(f"\033[91m The model can not load ckp from '{init_weight}' \033[0m")
            return
        
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms as pth_transforms
    from common import loadConfig, dumpPickle

    transform = pth_transforms.Compose([
        pth_transforms.Resize((352, 352)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    names = ["ILSVRC2012_test_00000003.jpg", "ILSVRC2012_test_00000025.jpg", "ILSVRC2012_test_00000023.jpg",
             "ILSVRC2012_test_00000086.jpg"]
    imgs = [Image.open(r"assets/datasets/DUTS/DUTS-TE/DUTS-TE-Image/{}".format(x)) for x in names]
    x = torch.stack([transform(img) for img in imgs], dim=0)

    cfg = loadConfig()
    net = Network(cfg)
    # net.loadCheckPoint(cfg.snapShot)
    net.train()
    # net.eval()
    out = net(x, global_step = 0.101)
    print("loss",out["loss"])
    print("pred", out["pred"].shape)