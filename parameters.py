#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:17:43 2019

@author: giuseppelisi
"""

import torch
import os
directory = os.path.join('..','digit_trajectories')

train_out_dir = os.path.join('..','train_out')
img_out_dir = os.path.join(train_out_dir,'img_out')
dev_out_dir = os.path.join(train_out_dir,'dev_out')
model_out_dir = os.path.join(train_out_dir,'model_out')

if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)
    
if not os.path.exists(dev_out_dir):
    os.makedirs(dev_out_dir)    
    
if not os.path.exists(model_out_dir):
    os.makedirs(model_out_dir)   


# Pythorch variables
if torch.cuda.is_available():
    torch_device = torch.device('cuda:0')
    using_cuda = True
else:
    torch_device = torch.device('cpu')   
    using_cuda = False
torch_dtype = torch.float32 

# DMP input
n_bfs = 50
n_dmps = 2
dt = .01
run_time = 1.

# Image generation
width = 28
height = 28