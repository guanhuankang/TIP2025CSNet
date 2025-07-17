#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import os
import sys
import argparse
import time
import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.enabled = False

from dataset import Data
from common import *
from network import Network
from loader import Loader
from testmodel import TestModel, main

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

def train(cfg):
    setlog(cfg=cfg)
    
    data   = Data(cfg, mode="train")
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batchSize, shuffle=True, pin_memory=True, num_workers=cfg.numWorkers)
    
    net    = Network(cfg)
    """ Uncomment this to load a full checkpoint
    for exmaple, when resume training from a middle checkpoint"""
    # net.loadCheckPoint(cfg.weights)
    net.train(True)
    net.cuda()
    
    logging.info("create optimizer and scheduler")
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weightDecay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.multi_lr_steps, gamma=cfg.gamma)
    sw = SummaryWriter(os.path.join(cfg.output_dir, cfg.eventPath))
    
    global_step = 0
    clock_begin = time.time()
    tot_iter = cfg.tot_iter
    
    datacfg = getattr(loadConfigByPath(cfg.datasetCfgPath), cfg.testSet)
    testResults = []

    while global_step < tot_iter:
        # optimizer.param_groups[0]['lr'] = (1.0 - (epoch / cfg.epoch)**0.9) * cfg.lr
        logging.info(f"tot_iter:{tot_iter} loader:{len(loader)}")
        net.train(True)
        for step, (image, mask) in enumerate(loader):
            optimizer.zero_grad()
            image, mask = image.cuda().float(), mask.cuda().float()
            if cfg.use_pseudo:
                loss = net(image, mask=mask, process=global_step/tot_iter)
            else:
                loss = net(image, mask=None, process=global_step/tot_iter)
            
            tot_loss = sum(v for v in loss.values())
            tot_loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
            
            ## log
            global_step += 1
            print_loss = dict((k, round(v.item(), 2)) for k,v in loss.items()) | {"total": tot_loss.item()}
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', print_loss, global_step=global_step)
            
            if step%10 == 0:
                elase = time.time() - clock_begin
                remain = elase/global_step * tot_iter - elase
                logging.info(f"steps:{global_step}/{tot_iter}, loss:{print_loss}, lr:{optimizer.param_groups[0]['lr']}, elase:{round(elase/60,1)}mins, eta:{round(remain/60, 1)}mins")
            if global_step > tot_iter:
                break
            
        if global_step > 0:
            ckp_path = os.path.join(cfg.output_dir, cfg.checkpointPath)
            latest_ckp = os.path.join(ckp_path, "model-{}-{}.pth".format(global_step, cfg.name))
            os.makedirs(ckp_path, exist_ok=True)
            torch.save(net.state_dict(), latest_ckp)
            
            with torch.no_grad():
                r = TestModel(resultPath=os.path.join(cfg.output_dir, cfg.evalPath)).test(datacfg=datacfg, model=net, name=cfg.name+str(global_step), checkpoint=None, crf=cfg.crf, save=True, size=cfg.size)
                sw.add_scalars("val", r.head(1).to_dict("records")[0], global_step=global_step)
                testResults.append({"iter": global_step, "name": cfg.name} | r.head(1).to_dict("records")[0])
                print(pd.DataFrame(testResults).set_index("iter").sort_index(), flush=True)
                logging.info(pd.DataFrame(testResults).set_index("iter").sort_index())
        ## end epoch test
    testResults = pd.DataFrame(testResults).set_index("iter").sort_index()
    testResults.to_csv(os.path.join(cfg.output_dir, cfg.name+"_results.csv"))
    logging.info(testResults)
    
    ## test on all datasets with post-process
    cfg.weights = latest_ckp
    main(cfg=cfg)


if __name__=='__main__':
    from config import loadCfg
    cfg = loadCfg(filename="config.json")
    train(cfg)