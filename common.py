import os, json, pickle
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf

# def getProjectRoot():
#     return os.path.abspath(os.path.join(os.getcwd(),".."))

# def getCurrentDirectory():
#     return os.path.abspath(os.getcwd())

def loadConfig(filename="config.json"):
    return loadConfigByPath(filename)

def loadConfigByPath(configPath):
    cvtkey = lambda x: dict( (k.replace("-","_"),v) for k,v in x.items() )
    with open(configPath, "r") as f:
        cfg = json.load(f, object_hook=lambda x: SimpleNamespace(**cvtkey(x)))
    return cfg

def loadJson(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def dumpPickle(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def nestedNameSpaceFromDict(dic):
    return json.loads(json.dumps(dic), object_hook=lambda x: SimpleNamespace(**x))

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def min2D(m):
    return torch.min(torch.min(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]


def max2D(m):
    return torch.max(torch.max(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]


def upx2(m):
    b, d, h, w = m.shape
    return F.interpolate(m, size=(2 * h, 2 * w), mode="bilinear")


def minMaxNorm(m, eps=1e-12):
    return (m - min2D(m)) / (max2D(m) - min2D(m) + eps)