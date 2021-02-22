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

import os
import sys

from tqdm.auto import tqdm, trange

import argparse

import json

def exists(path_str):
    return os.path.exists(path_str)

def canread(pname):
    if not os.path.exists(pname):
        raise FileNotFoundError(pname)
    if not os.access(pname, os.R_OK):
        raise PermissionError(pname)
    return True

def readable_path(pname):
    canread(pname)

    if not os.path.isdir(pname):
        raise NotADirectoryError(pname)

    return pname

def readable_file(fname):
    canread(fname)

    if not os.path.isfile(fname):
        raise IsADirectoryError(fname)

    return fname

noisekeys = ["awn", "srn", "drop", "clean"]

parser = argparse.ArgumentParser(description="Script for testing with SimpleCNN classifier.")

# Base directory of the Intel Image Classification database.
parser.add_argument('--db_path',
                    type=readable_path,
                    help="The base path of the Intel Image Classification database.",
                    metavar="/base/database/path",
                    required=True
                    )

# CNN Classifier filename.
parser.add_argument('--clf_fname',
                    type=readable_file,
                    help="File name of the torch classifier.",
                    required=True
                    )

# CDL filename.
parser.add_argument('--cdl_fname',
                    type=readable_file,
                    default=None,
                    help="File name of the convolutional dictionary. If none is given, " + \
                        "the classifier will not use CD to reconstruct the data."
                    )

# Noise profile options
parser.add_argument('--is_train', "-tr",
                    action="store_true",
                    help="Use the training set. (Default: %(default)s)"
                    )

# Noise profile options
parser.add_argument('--noise_config_fname', "-n",
                    type=readable_file,
                    default=None,
                    help="Configuration file of noise profiles. If no filename is provided, %(prog)s will run " + \
                        "without any noise."
                    )

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    if args.noise_config_fname:
        with open(args.noise_config_fname, 'r') as fobj:
            noise_config = json.load(fobj)
    else:
        noise_config = None
    print(args)
    print(unknown)
    print(noise_config)
    for k in noisekeys:
        noise_list = noise_config.get(k, [])
        print(noise_config.get(k, []))
        print(len(noise_list))