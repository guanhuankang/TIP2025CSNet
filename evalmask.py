#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import os
import sys
import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import progressbar
import pandas as pd
from net.utils import CRF
from common import loadConfig


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


class EvalModel:
    def __init__(self, resultPath="output"):
        self.resultPath = resultPath
        os.makedirs(resultPath, exist_ok=True)
        self.results = []
        self.indexs = []
        self.crf = CRF()

    def eval(self, pred, mask):
        pred = F.interpolate(pred, size=mask.shape[2::], mode="bilinear")
        assert pred.shape==mask.shape, "pred shape:{}, mask shape:{}".format(pred.shape, mask.shape)
        pos = pred.gt(0.5).float()
        tp = (pos * mask).sum()
        prc = tp / (pos.sum()+1e-6)
        rec = tp / (mask.sum()+1e-6)

        mae = torch.abs(pred-mask).mean()
        fbeta = 1.3*(prc * rec) / (0.3 * prc + rec + 1e-9)
        acc = (pos == mask).sum() / torch.numel(mask)
        iou = tp / (pos.sum() + mask.sum() - tp + 1e-6)
        gtp = mask.sum() / torch.numel(mask)
        predp = pos.sum() / torch.numel(pred)
        result = {
            "mae": mae.item(),
            "fbeta": fbeta.item(),
            "acc": acc.item(),
            "iou": iou.item(),
            "gtp": gtp.item(),
            "predp": predp.item()
        }
        return result

    def record(self, name, result):
        self.results.append(result)
        self.indexs.append(name)

    def report(self, name):
        resultPd = pd.DataFrame(self.results, index=self.indexs)
        resultPd = pd.concat([resultPd.agg("mean").to_frame(name="mean").T, resultPd], ignore_index=False)
        resultPd.to_csv(os.path.join(self.resultPath, name+".csv"))
        print(resultPd.head(1), flush=True)
        return resultPd

    def clear(self):
        self.results = []
        self.indexs = []

    def test(self, datacfg, results, name="test", crf=0, save=True, size=(352,352)):
        name_list = [x[0:-len(datacfg.image.suffix)] for x in os.listdir(os.path.join(datacfg.root, datacfg.image.path)) if x.endswith(datacfg.image.suffix)]
        widgets = ["[",progressbar.Timer(),"]",progressbar.Bar("*"),"(",progressbar.ETA(),")"]
        bar = progressbar.ProgressBar(maxval=len(name_list), widgets=widgets).start()
        
        with torch.no_grad():
            for i,name in enumerate(name_list):
                img = np.array(Image.open(os.path.join(datacfg.root, datacfg.image.path, name+datacfg.image.suffix)).convert("RGB"))
                mak = np.array(Image.open(os.path.join(datacfg.root, datacfg.mask.path, name+datacfg.mask.suffix)).convert("L")).astype(float) / 255.0
                mak = torch.tensor(mak).unsqueeze(0).unsqueeze(0).gt(0.5).float()  ## 1, 1, H, W

                ## 1, 1, H, W
                pred = torch.from_numpy(np.array( Image.open(os.path.join(results, name+".png")).convert("L") ).astype(float) / 255.).unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred, size=mak.shape[2::], mode="bilinear")
                pred_refine = pred
                if crf>0:
                    img = F.interpolate(torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2), size=mak.shape[2::], mode="bilinear") / 255.
                    pred_refine = self.crf( img, pred, iters=crf )
                if crf<0:
                    from bilateral_solver import bilateral_solver_output
                    # img = F.interpolate(torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2), size=mak.shape[2::], mode="bilinear")
                    out_solver, binary_solver = bilateral_solver_output(None, np.array(pred[0][0]), img)
                    pred_refine = torch.from_numpy(binary_solver).float().unsqueeze(0).unsqueeze(0)
                if torch.abs(pred_refine - pred).mean() > torch.abs((1.0 - pred_refine) - pred).mean():
                    pred = 1.0 - pred_refine
                else:
                    pred = pred_refine
                
                result = self.eval(pred, mak)
                self.record(name, result)
                if save:
                    os.makedirs(self.resultPath, exist_ok=True)
                    Image.fromarray((pred[0,0].numpy()*255).astype(np.uint8)).save(os.path.join(self.resultPath,name+".png"))
                bar.update(i)

            bar.finish()
        
        rep = self.report(name)
        self.clear()
        return rep

def main(cfg):
    setlog(cfg=cfg)
    
    datacfg = loadConfig(cfg.datasetCfgPath)
    
    results = []
    indexs = []
    dname_lst = [cfg.testSet] if isinstance(cfg.testSet, str) else cfg.testSet
    for dname in dname_lst:
        EM = EvalModel(resultPath=os.path.join(cfg.output_dir, cfg.evalPath, dname))
        r = EM.test(
            datacfg=getattr(datacfg, dname), 
            results=cfg.tobe_eval_path,
            name=dname+"_"+cfg.name, 
            crf=cfg.crf_round, 
            save=cfg.save | False, 
            size=cfg.size
        )
        results.append( r.head(1).to_dict("records")[0] )
        indexs.append(dname)
        pdR = pd.DataFrame(results, index=indexs)
        pdR.to_csv(os.path.join(cfg.output_dir, cfg.evalPath, "results.csv"))
        print(pdR, flush=True)

if __name__=='__main__':
    from config import loadCfg
    cfg = loadCfg(filename="config.json")

    ## Our full results (selfmask as detector)
    result_path = "assets/our_results/ours_full_without_crf/inference"
    for ds in ["ECSSD", "DUTS", "HKU_IS", "PASCAL_S", "DUT_OMRON", "SOC"]:
        cfg.tobe_eval_path = os.path.join(result_path, ds)
        cfg.testSet = ds
        main(cfg)
        