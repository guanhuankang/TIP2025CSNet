#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import os
import argparse
import json
import sys
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as pth_transforms
import progressbar
import time

from thop import profile
from thop import clever_format

from net.utils import CRF
from common import loadConfig
from network import Network


def setlog(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = os.path.join(cfg.output_dir, f"{cfg.name}.log")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True  # optional: reset any prior basicConfig; remove if undesired
    )

class TestModel:
    def __init__(self, resultPath="output"):
        self.resultPath = resultPath
        os.makedirs(resultPath, exist_ok=True)
        self.crf = CRF()

    def test(self, datacfg, model, checkpoint=None, name="test", crf=0, save=False, size=(352,352)):
        logging.info(f"{name}: ckp={checkpoint}, crf:{crf}, save:{save}")
        
        name_list = [x[0:-len(datacfg.image.suffix)] for x in os.listdir(os.path.join(datacfg.root, datacfg.image.path)) if x.endswith(datacfg.image.suffix)]
        transform = pth_transforms.Compose([
            pth_transforms.Resize(size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        widgets = ["[",progressbar.Timer(),"]",progressbar.Bar("*"),"(",progressbar.ETA(),")"]
        bar = progressbar.ProgressBar(maxval=len(name_list), widgets=widgets).start()
        
        model.loadCheckPoint(checkpoint)    
        model.eval()
        model.train(False)
        
        rtime = 0.0
        crftime = 0.0
        cnt = 0
        with torch.no_grad():
            for i,name in enumerate(name_list):
                img = transform(Image.open(os.path.join(datacfg.root, datacfg.image.path, name+datacfg.image.suffix)).convert("RGB")).unsqueeze(0)
                
                flops, params = profile(model, inputs=(img.cuda(),), verbose=False)
                flops, params = clever_format([flops, params], "%.3f")
                logging.info(f"FLOPs: {flops}, Parameters: {params}")
                
                stime = time.time_ns()
                pred = model(img.cuda())["pred"].cpu()
                etime = time.time_ns()
                rt = etime - stime
                rtime += rt
                cnt += 1
                logging.info(f"Avg Runtime: {round(rtime / cnt / 1e6, 3)} ms [count on {cnt} samples.]")
                
                # ori_img = self.crf.recover(img)
                # stime = time.time_ns()
                # img = F.interpolate(img, size=pred.shape[2::], mode="bilinear")
                # pred_refine = self.crf( ori_img, pred, iters=crf )
                # etime = time.time_ns()
                # rt = etime - stime
                # crftime += rt
                # print(f"CRF time ({crf}): {round(crftime / cnt / 1e6, 3)} ms.")
                    
                bar.update(i)

            bar.finish()

def main(cfg):
    setlog(cfg=cfg)
    
    datacfg = loadConfig(cfg.datasetCfgPath)
    net = Network(cfg).cuda()
    net.model.process = 1.0
    
    dname_lst = ["ECSSD", ]
    # dname_lst = ["ECSSD", "DUT_OMRON", "SOD", "SOC_AC", "SOC_BO", "SOC_CL", "SOC_HO", "SOC_MB", "SOC_OC", "SOC_OV", "SOC_SC", "SOC_SO", "SOC"]
    for dname in dname_lst:
        testModel = TestModel(resultPath=os.path.join(cfg.output_dir, cfg.evalPath, dname))
        r = testModel.test(
            datacfg=getattr(datacfg, dname), 
            model=net, 
            name=dname+"_"+cfg.name, 
            checkpoint=cfg.weights, 
            crf=cfg.crf_round, 
            save=True, 
            size=cfg.size
        )

if __name__=='__main__':
    from config import loadCfg
    cfg = loadCfg(filename="config.json")
    main(cfg)