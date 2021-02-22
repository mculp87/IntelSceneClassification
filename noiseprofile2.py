#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:52:49 2021

@author: mculp
"""

import numpy as np

import warnings

DBG_SEED = 2718


class NoiseProfile:
    def __init__(self, seed=None, size=None, dtype=np.float32):
        '''
        Parameters
        ----------
        seed : int or 1-d array_like, optional
            Initial seed for PRNG engine. The default is None.
        size : TYPE, optional
            DESCRIPTION. The default is None.
        dtype : TYPE, optional
            DESCRIPTION. The default is np.float32.

        Returns
        -------
        None.

        '''
        self.seed = seed if seed else DBG_SEED

        np.random.seed(self.seed)

        self.epsilon = self.sample(size=size, dtype=dtype) if size else None

    def sample(self, size=None, dtype=np.float32):
        raise NotImplementedError()

    def transform(self, X, idx, *, out=None):
        raise NotImplementedError()


class DropoutNoise(NoiseProfile):
    def __init__(self, mean, *, size=None, seed=None, dtype=np.bool):
        # assert 0.0 <= mean <= 1.0, "The mean dropout rate must be in [0,1]."
        self.mean = mean

        if dtype not in (np.bool, bool):
            wrnmsg = "DropoutNoise does not support non-boolean types." + \
                " Non-boolean type will be ignored."
            warnings.warn(wrnmsg)
        
        super().__init__(seed=None, size=size, dtype=np.bool)

    def sample(self, size=None, dtype=np.bool):
        self.size = size if size else self.size
        self.epsilon = np.random.binomial(1, self.mean, self.size).astype(np.bool)
        return self.epsilon

    def transform(self, X, idx, out=None):
        return np.multiply(X, self.epsilon[idx, ...], out=out)

