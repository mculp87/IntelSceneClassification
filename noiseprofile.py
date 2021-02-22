#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:52:49 2021

@author: mculp
"""

import numpy as np

import warnings

import codecs, json

DBG_SEED = 2718

# Creating a metaclass so whenever a child class of NoiseProfile is instantiate, there is a new
# instantiation for NoiseProfile. Meaning each instance will have its own copy of the parent class
# variables. To understand this further, look up python metaclass variables and how __call__, __init__
# __new__ all work with respect to 'object' and 'type'.
class Metaclass(type):    
    def __new__(cls, name, bases, dct):
        return super().__new__(cls, name, bases, dct)

class NoiseProfile(metaclass=Metaclass):
    # def __init__(cls, seed=DBG_SEED, size=None, dtype=np.float32):
    def __init__(self, param, seed=DBG_SEED, size=None, dtype=np.float32):
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
        self.param = param
        self.seed = seed
        self.size = size
        self.dtype = dtype

        np.random.seed(self.seed)

        self.epsilon = self.sample(size=size, dtype=dtype) if size else None
    
    @classmethod
    def load(cls, fname):
        npfile = np.load(fname, allow_pickle=True)
        x = super().__new__(cls)
        nonarrayvars = ["param", "seed", "size"]
        for k in nonarrayvars:
            # need to cast out of the numpy array
            setattr(x, k, npfile[k][()])
        setattr(x, "epsilon", npfile["epsilon"])
        x.dtype = x.epsilon.dtype
        return x
    
    def save(self, fname):
        fdct = {
            "param": self.param,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "size": self.size
        }
        np.savez_compressed(fname, **fdct)
    
    def loadsample(self, fname):
        try:
            epsilon = np.load(fname)
        except Exception as e:
            print(e.message, e.args)
            epsilon = None
        self.epsilon = epsilon

    def sample(self, size=None, dtype=np.float32):
        raise NotImplementedError()

    def transform(self, X, idx, *, out=None):
        raise NotImplementedError()

    def __str__(self):
        return f"$p\\left(\\theta = {self.param}\\right)$"


class AdditiveWhiteNoise(NoiseProfile):
    def sample(self, size=None, dtype=np.float32):
        self.size = size if size else self.size
        self.epsilon = np.random.normal(self.param["mean"],
                                        self.param["std"],
                                        self.size).astype(dtype)
        return self.epsilon

    def transform(self, X, idx, *, out=None):
        return np.add(X, self.epsilon[idx, ...], out=out)

    def __init__(self, param, **kwargs):
        assert isinstance(param, dict), "Parameter argument must be a dictionary."

        super().__init__(param, **kwargs)

        # Set default values to unit normal if none are provided.
        self.param["mean"] = self.param.get("mean", 0.0)
        self.param["std"] = self.param.get("std", 1.0)
        
        assert 0.0 <= self.param["std"], "Standard deviation must be non-negative."
        

    def __str__(self, *, paramstr=None):
        if paramstr:
            return f"Normal$\\left({paramstr}\\right)$"
        else:
            return f"Normal$\\left({self.param['mean']}, {self.param['std']}\\right)$"


class ZeroNoise(NoiseProfile):
    def sample(self, size=None, dtype=np.float32):
        self.size = size if size else self.size
        return np.zeros(self.size, dtype=dtype)

    def transform(self, X, idx, *, out=None):
        if out:
            out = X
        else:
            return X

    def __init__(self, *, size=None, seed=None, dtype=np.float32):
        super().__init__(seed=seed, size=size, dtype=dtype)

    def __str__(self):
        return f"p$\\left(X = 0\\right) = 1$"

class ScalarRayleighNoise(NoiseProfile):
    def sample(self, size=None, dtype=np.float32):
        self.size = size if size else self.size
        self.epsilon = np.random.rayleigh(scale=self.param["scale"],
                                          size=self.size).astype(dtype)
        return self.epsilon

    def transform(self, X, idx, *, out=None):
        return np.multiply(X, self.epsilon[idx, ...], out=out)

    def __init__(self, param, **kwargs):
        assert isinstance(param, dict), "Parameter argument must be a dictionary."

        super().__init__(param, **kwargs)

        self.param["scale"] = self.param.get("scale", 1.0)

        assert 0.0 < self.param["scale"], "Rayleigh scale parameter must be positive."


    def __str__(self, *, paramstr=None):
        if paramstr:
            return f"Rayleigh$\\left({paramstr}\\right)$"
        else:
            return f"Rayleigh$\\left({self.param['scale']:.4f}\\right)$"


class DropoutNoise(NoiseProfile):
    def __init__(self, param, **kwargs):
        assert isinstance(param, dict), "Parameter argument must be a dictionary."

        self.axis = kwargs.get("axis", None)

        kwargs["dtype"] = kwargs.get("dtype", np.bool)
        if kwargs["dtype"] not in (np.bool, bool):
            wrnmsg = "DropoutNoise does not support non-boolean types." + \
                " Non-boolean type will be ignored."
            warnings.warn(wrnmsg)
            self.dtype = np.bool

        super().__init__(param, **kwargs)


        self.param["rate"] = self.param.get("rate", 1.0)

        assert 0.0 <= self.param["rate"] <= 1.0, "The mean dropout rate must be in [0,1]."


    def sample(self, size=None, dtype=np.bool):
        self.size = size if size else self.size
        if self.axis is None:
            self.epsilon = np.random.binomial(1, 1 - self.param["rate"], self.size).astype(np.bool)
            return self.epsilon
        
        copyaxis = list(self.axis) if isinstance(self.axis, tuple) else [self.axis]
        workaxis = list([n for n in np.arange(len(self.size)) if n not in copyaxis])

        copyshape = np.array(self.size, dtype=np.uint)[copyaxis]
        workshape = np.array(self.size, dtype=np.uint)[workaxis]
        tempshape = np.concatenate([workshape, copyshape])
        
        eps = np.random.binomial(1, 1.0 - self.param["rate"], tuple(workshape)).astype(np.bool)
        eps = np.repeat(eps, np.prod(copyshape)).reshape(tempshape)
        self.epsilon = np.moveaxis(eps,
                    source=tuple(np.arange(len(tempshape))[slice(-len(copyshape),None)]),
                    destination=self.axis)
        return self.epsilon

    def transform(self, X, idx, out=None):
        return np.multiply(X, self.epsilon[idx, ...], out=out)
    
    def __str__(self, *, paramstr=None):
        if paramstr:
            return f"Bernoulli$\\left({paramstr}\\right)$"
        else:
            return f"Bernoulli$\\left({1.0 - self.param['rate']:.4f}\\right)$"
    
# class DatarotNoise(DropoutNoise):
#     def sample(self, size=None, dtype=np.bool):
#         self.size = size if size else self.size
        
#         # Binary coefficients, used to convert extended tensor of the boolean matrix into the uint8.
#         b = np.power(2, np.range(8, dtype=np.uint8), dtype=np.uint8)

#         if self.axis is None:
#             eps = np.random.binomial(1, self.rate, self.size).astype(np.uint8)
#             self.epsilon = np.einsum('...k,k->...', E, b, dtype=np.uint8)
#             return self.epsilon
        
#         copyaxis = list(self.axis) if isinstance(self.axis, tuple) else [self.axis]
#         workaxis = list([n for n in np.arange(len(self.size)) if n not in copyaxis] + (-1,))

#         copyshape = np.array(self.size, dtype=np.uint)[copyaxis]
#         workshape = np.array(self.size, dtype=np.uint)[workaxis]
#         tempshape = [workshape, copyshape]
#         print(copyshape)
        
#         eps = np.random.binomial(1, self.rate, tuple(workshape) + (8,)).astype(np.uint8)
#         eps = np.einsum('...k,k->...', eps, b, out=eps)
#         eps = np.repeat(eps, np.prod(copyshape)).reshape(tempshape)
#         # print(np.arange(len(tempshape))[slice(-len(copyshape),None)])
#         self.epsilon = np.moveaxis(eps,
#                     source=tuple(np.arange(len(tempshape))[slice(-len(copyshape),None)]),
#                     destination=self.axis)
#         return self.epsilon
    
#     def transform(self, X, idx, out=None):
#         return np.divide(np.bitwise_xor(np.uint8(255*X), self.epsilon), 255.0, out=out)

#     def __str__(self, *, paramstr=None):
#         if paramstr:
#             return f"$\\sum_{n} 2^n$ Bernoulli$\\left({paramstr}\\right)$"
#         else:
#             return f"$\\sum_{n} 2^n$ Bernoulli$\\left({self.rate:.4f}\\right)$"

# class ChannelDropoutNoise(NoiseProfile):
#     def __init__(self, channel, *, size=None, seed=None, dtype=np.bool):
#         if not isinstance(channel, tuple):
#             self.channel = [channel]
#         else:
#             self.channel = list(channel)

#         if dtype not in (np.bool, bool):
#             wrnmsg = "DropoutNoise does not support non-boolean types." + \
#                 " Non-boolean type will be ignored."
#             warnings.warn(wrnmsg)
        
#         super().__init__(seed=None, size=size, dtype=np.bool)

#     def sample(self, size=None, *, dtype=np.bool):
#         self.size = size if size else self.size
#         self.epsilon = np.full(self.size, True, dtype=np.bool)
#         self.epsilon[:,self.channel,...] = False
#         return self.epsilon

#     def transform(self, X, idx, out=None):
#         return np.multiply(X, self.epsilon[idx, ...], out=out)

