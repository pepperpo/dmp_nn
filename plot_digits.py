#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:07:31 2019

@author: giuseppelisi
"""

import matplotlib.pyplot as plt
from parameters import directory
from utils import load_trajectories,plot_multi_stroke,generate_trajectories_from_template,plot_ode_output,traj2img,affine_transform,plot_res,apply_affine
from parameters import n_bfs,dt,run_time,torch_device,torch_dtype,dev_out_dir,width,height
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
import torch
from torchvision.utils import save_image
import os
import matplotlib.image as mpimg



trajectories = load_trajectories(directory)

n_traj = len(trajectories)
plt.figure(figsize=(2*n_traj,2))
for i_k,key in enumerate(sorted(trajectories.keys())):
    plt.subplot(1,n_traj,i_k+1)
    traj = trajectories[key]
    plot_multi_stroke(traj,key)




batch_size_per_digit = 8
all_keys = sorted(trajectories.keys())
traj,batch_t,true_class = generate_trajectories_from_template(batch_size_per_digit,n_bfs,dt,run_time,torch_device,torch_dtype)
traj = apply_affine(traj,torch_device,torch_dtype)
X_plot = traj2img(traj,width,height,torch_dtype,torch_device,rbf_w=None)



train_dataset = datasets.MNIST(root=os.path.join('..','data'),
            train=True,
            download=False,
            transform=transforms.ToTensor())
    
train_loader = DataLoader(train_dataset, shuffle=True,
        batch_size=80)

batch_i=0
mnist_data, mnist_target = next(itertools.islice(train_loader, batch_i, None))
mnist_data, mnist_target = mnist_data.to(torch_device), mnist_target.to(torch_device).long()

#real_img1 = plot_res(traj,X_plot)


real_img = torch.cat([X_plot.unsqueeze(1).cpu(),mnist_data.cpu()])


f_save = os.path.join(dev_out_dir,'plot_img.png')
save_image(real_img,f_save)
plt.figure(figsize=(6,10))
img=mpimg.imread(f_save)
plt.imshow(img)


plt.show()

    



