#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:16:09 2021

@author: mculp
"""

import numpy as np

from PIL import Image
import torch

import os
import glob

import noiseprofile as noise


class IntelSceneDataset(torch.utils.data.Dataset):
    def applyNoise(self, func):
        return func()

    def getFolderList(self, path):
        return [fname for fname in os.listdir(path)
                if os.path.isdir(os.path.join(path, fname))]

    def getFileList(self, path):
        return [fname for fname in os.listdir(path)
                if os.path.isfile(os.path.join(path, fname))]

    def __init__(self, basepath, *, segment="train", transform=None):
        self.transform = transform
        self.basepath = basepath
        if segment == "train":
            segpath = os.path.join(self.basepath, "seg_train/seg_train")
            self.categories = self.getFolderList(segpath)
        elif segment == "test":
            segpath = os.path.join(self.basepath, "seg_test/seg_test")
            self.categories = self.getFolderList(segpath)

        labelpath = {label: os.path.join(segpath, label)
                     for label in self.categories}
        imgfnames = {label: glob.glob(os.path.join(path, "*.jpg"))
                     for label, path in labelpath.items()}
        self.fnames = [fname for cfnames in imgfnames.values()
                       for fname in cfnames]

        self.categories = list(imgfnames.keys())

        # Create dimension size parameters.
        self.numSamples = np.uint(sum(len(lst) for lst in imgfnames.values()))
        self.numChannels = np.int64(3)
        
        # Fixed image dimension for Intel dataset (could be changed).
        self.imgsizes = (np.int64(150), np.int64(150))
        
        # Using shape consistent with Torch tensor convention.
        self.shape = (self.numSamples, self.numChannels) + self.imgsizes
        self.y = torch.zeros((self.numSamples,),
                             dtype=torch.int64)

        cumNum = np.zeros((len(self.categories) + 1,), dtype=np.uint)
        cumNum[1:] = np.cumsum([len(fnames) for fnames in imgfnames.values()])
        for c in range(len(self.categories)):
            self.y[cumNum[c]:cumNum[c+1]] = c

    def __len__(self):
        return self.numSamples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.fnames[idx]
        npimg = np.array(Image.open(fname).convert(mode="RGB"), np.uint)
        Psig = np.linalg.norm(npimg.ravel(), ord=2)

        if self.transform:
            img = self.transform(Image.open(fname).convert(mode="RGB"))
        else:
            img = Image.open(fname).convert(mode="RGB")

        return {"images": img,
                "labels": self.y[idx],
                "Psig": Psig,
                "Pnoi": 1.0,
                "idx": idx}
    
    def numCategories(self):
        return len(self.categories)


class IntelNoiseSceneDataset(IntelSceneDataset):
    '''
    Wrapper of IntelSceneDataset.
    '''
    def __init__(self,
                 dataset,
                 noiseprofile,
                 *,
                 transform=None,
                 noisefname=None):
        self.dataset = dataset
        
        self.noiseprofile = noiseprofile
        if self.noiseprofile.epsilon is None:
            if noisefname is None:
                self.noiseprofile.sample(self.dataset.shape)
            else:
                self.noiseprofile.loadsample(noisefname)
        pass

    @classmethod
    def newinstance(cls, datapath, noiseprofile, **kwargs):
        noisedataset = cls(IntelSceneDataset(datapath, **kwargs), noiseprofile)
        return noisedataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.dataset.fnames[idx]

        # Copy pil image into ndarray, scaled to [0,1] floats.
        pilimg = Image.open(fname).convert(mode="RGB").resize(self.dataset.imgsizes)
        npimg = np.array(pilimg, dtype=np.float32).transpose((2, 0, 1))

        # $P_{\text{sig}} = \| X_{\text{sig}} \|$
        truimg = npimg.copy()
        Psig = np.linalg.norm(truimg.ravel(), ord=2)

        npimg /= 255.0

        np.clip(npimg, 0, 1, out=npimg)

        # In place noise transformation of the ndarray.
        self.noiseprofile.transform(npimg, idx, out=npimg)

        np.clip(npimg, 0, 1, out=npimg)

        # Rescale to 255 intensity values.
        npimg *= 255.0

        # SNR = ||signal|| / ||signal - transform(signal)||
        Pnoise = np.linalg.norm((truimg - npimg).ravel(), ord=2)

        # Convert back to PIL image, cause pytorch collation hates numpy.
        pilimg = Image.fromarray(npimg.transpose((1, 2, 0)).astype(np.uint8))

        if self.dataset.transform:
            img = self.dataset.transform(pilimg)
        else:
            img = pilimg

        return {"images": img,
                "labels": self.dataset.y[idx],
                "Psig": Psig,
                "Pnoi": Pnoise,
                "idx": idx}

    def __len__(self):
        return len(self.dataset)