#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:37:42 2021

@author: mculp
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm.auto import tqdm, trange


class SimpleCNN(nn.Module):
    def __init__(self, numClasses):
        super(SimpleCNN, self).__init__()
        assert numClasses > 1, "Must have 2 or more classes."
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 34 * 34, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, numClasses)
        self.numClasses = numClasses

    def forward(self, x):
        # relu preserves dimensions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, np.prod(x.shape[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleClassifier:
    # Default device options
    deviceopt = {
        "device": torch.device("cpu"),
        "non_blocking": True
    }
    def __init__(self, categories, network=None, *, fname=None):
        if isinstance(categories, int) and categories > 1:
            self.categories = range(categories)
        elif isinstance(categories, list):
            self.categories = categories
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.deviceopt["device"] = self.device

        self.fname = fname
        if network:
            self.network = network
        elif fname:
            self.load()
        else:
            self.network = SimpleCNN(len(self.categories)).to(self.device,
                                    non_blocking=True,
                                    dtype=torch.float32)

        self.metric = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.network.parameters(),
                                    lr=0.001,
                                    momentum=0.9)

    def train(self, dataloader, *,
            epochs=10,
            verbose=1,
            calcsnr=True,
            window=100,
            preprocessor=None):
        for epoch in tqdm(range(epochs), desc="Training epoch"):
            cumloss = 0.0
            if calcsnr:
                snr = [None]*len(dataloader)
                tdesc = "[%d] (Loss, SNR): (%.4f, %f)" % (0, 0.0, 0.0)
                cumsnr = 0
            else:
                tdesc = "[%d] Loss: %.4f" % (0, 0.0)
    
            t = tqdm(enumerate(dataloader, 0),
                     desc=tdesc,
                     total = len(dataloader),
                     leave = False)
            for n, data in t:
                # y_pred = NN()
                X = data["images"].cuda(non_blocking=True)
                y = data["labels"].cuda(non_blocking=True)
                if preprocessor is not None:
                    X = preprocessor(X)

                if calcsnr:
                    Psignal = data["Psig"]
                    Pnoise = data["Pnoi"]
                    snr[n] = 10.0 * np.log10(Psignal / Pnoise)
                    cumsnr += torch.sum(snr[n])

                self.optimizer.zero_grad()
                z = self.network(X)
                loss = self.metric(z, y)
                loss.backward()
                self.optimizer.step()

                cumloss += loss.item()
                if (n + 1) % window == 0:
                    self.save()

                    if calcsnr:
                        tlab = "[%d] (Short-term Loss, SNR): (%.4f, %f)"
                        tval = (n + 1, cumloss / window, cumsnr / window)
                        cumsnr = 0.0
                    else:
                        tlab = "[%d] Short-term Loss: %.4f"
                        tval = (n + 1, cumloss / window)
                    tdesc = tlab % tval
                    t.set_description(tdesc)
                    cumloss = 0.0

    def save(self, f=None):
        if f:
            torch.save(self.network, f)
        elif self.fname:
            torch.save(self.network, self.fname)
        else:
            error("No filename has been provided.")

    def load(self, f=None):
        if f:
            self.network = torch.load(f)
        elif self.fname:
            self.network = torch.load(self.fname)
        else:
            error("No filename has been provided.")
        return self.network

    def predict(self, dataloader, *, calcsnr=True, window=100, preprocessor=None):
        C = np.zeros((len(self.categories), len(self.categories)), dtype=np.uint)
        
        if not calcsnr:
            snr = None
        
        cnt = 0
        if calcsnr:
            snr = [None]*len(dataloader)
            tdesc = "[%d] (Acc, SNR): (%.4f, %f)" % (0, 0.0, 0.0)
            cumsnr = 0
        else:
            tdesc = "[%d] Accuracy: %.4f" % (0, 0.0)
        
        labelpred = -np.ones(len(dataloader.dataset))

        t = tqdm(enumerate(dataloader, 0),
                 desc = tdesc,
                 total = len(dataloader),
                 leave=False)
        with torch.no_grad():
            for n, data in t:
                X = data["images"].cuda(non_blocking=True)
                ytrue = data["labels"].cuda(non_blocking=True)
                idx = data["idx"]
                if preprocessor is not None:
                    X = preprocessor(X)

                cnt += X.shape[0]

                _, ypred = torch.max(self.network(X.cuda(non_blocking=True)), 1)
                labelpred[idx] = ypred.cpu()
                
                for m in range(ypred.shape[0]):
                    C[ytrue[m], ypred[m]] += np.uint(1)

                if calcsnr:
                    Psignal = data["Psig"]
                    Pnoise = data["Pnoi"]
                    snr[n] = 10.0 * np.log10(Psignal / Pnoise)
                    cumsnr = torch.sum(snr[n])

                if (n + 1) % window == 0:
                    accuracy = np.sum(np.diag(C)) / np.sum(C)

                    if calcsnr:
                        meansnr = torch.mean(torch.cat(snr[:n+1]))
                        tdesc = "[%d] (Acc, SNR): (%.4f, %f)" % (n + 1, accuracy, meansnr)
                    else:
                        tdesc = "[%d] Accuracy: %.4f" % (n + 1, accuracy)
                    t.set_description(tdesc)
        snr = np.concatenate(snr) if calcsnr else None
        return {"Conf": C,
                "labels": labelpred,
                "SNR": snr}