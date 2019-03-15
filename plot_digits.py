#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:07:31 2019

@author: giuseppelisi
"""

import matplotlib.pyplot as plt
from parameters import directory
from utils import load_trajectories

trajectories = load_trajectories(directory)

n_traj = len(trajectories)
plt.figure(figsize=(18,2))
for i_k,key in enumerate(sorted(trajectories.keys())):
    plt.subplot(1,n_traj,i_k+1)
    traj = trajectories[key]
    plt.plot(traj[:,0],traj[:,1])
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(key)
    plt.axis('off')
plt.show()




