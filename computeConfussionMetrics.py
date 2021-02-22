%reload_ext autoreload
%autoreload 2

import numpy as np

import torch
import torchvision as tv

import noiseprofile as noise
import IntelSceneDataset as isds
import classifiers
from classifiers import SimpleClassifier

import denoise

from tqdm.auto import tqdm

from sporco import util
from sporco import linalg
from sporco import plot
plot.config_notebook_plotting()
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.dictlrn import onlinecdl
from sporco.cupy.admm import cbpdn
from sporco.cupy import cnvrep
from sporco.cupy import linalg as cplinalg
from sporco.cupy import prox as cpprox
from sporco.cupy.linalg import irfftn,rfftn
from sporco.cupy.linalg import inner

import os

import matplotlib.pyplot as plt

outpath = "./data/out/"
noisepath = os.path.join(outpath, "noise")
clfpp = "PP"
clfname = f"Resnet50CNN_{clfpp}"
confpath = os.path.join(outpath, "conf")

dbpath = "/mnt/hd-storage/IntelImageClassification"

transform = tv.transforms.Compose(
    [tv.transforms.CenterCrop(150),
     tv.transforms.ToTensor(),
     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cleanTe = isds.IntelSceneDataset(dbpath, segment="test", transform=transform)

noisetypes = ["awn", "srn", "drop"]