#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:07:31 2019

@author: giuseppelisi
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from parameters import directory

if len(sys.argv) != 2:
    raise Exception('Please provide the name of the digit in the command line')

if not os.path.exists(directory):
    os.makedirs(directory)

plt.figure()
plt.plot([])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Press enter to terminate')
tought_traj = np.array(plt.ginput(-1))
plt.close()

plt.figure()
plt.plot(tought_traj[:,0],tought_traj[:,1])
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Generated trajectory')
plt.show()


filename = os.path.join(directory,'{}_traj'.format(sys.argv[1]))
np.savez(filename,tought_traj)  


