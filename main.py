#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:52:42 2019

@author: giuseppelisi
"""
import os
import numpy as np

from utils import generate_trajectories_from_template,plot_res,apply_affine,traj2img

import torch
from dmp_nn import DMPLearn, ModelD
from torch import nn, optim

from torchvision.utils import save_image

from torchdiffeq import odeint_adjoint 
from torchdiffeq import odeint

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import itertools

import torch.nn.functional as F

from parameters import n_bfs,dt,run_time,torch_device,torch_dtype,model_out_dir,img_out_dir,n_dmps,width,height

batch_size = 256#64#256
epochs = 250


model_d = ModelD(n_bfs*n_dmps+n_dmps*2)
model_d.to(torch_device)
optim_d = optim.SGD(model_d.parameters(), lr=0.01)

t_zeros = torch.zeros(batch_size,n_dmps,device=torch_device,dtype=torch_dtype)
t_ones = torch.ones(batch_size,n_dmps,device=torch_device,dtype=torch_dtype)

print_every_itr = 25
save_every = 1

train_dataset = datasets.MNIST(root='../data',
            train=True,
            download=False,
            transform=transforms.ToTensor())
    
train_loader = DataLoader(train_dataset, shuffle=True,
        batch_size=batch_size)


model_d.train()
for epoch_idx in range(epochs):
    
    pred_y,batch_t,true_class = generate_trajectories_from_template(2560,n_bfs,dt,run_time,torch_device,torch_dtype)
    Y_train = pred_y.cpu().numpy()
    n_samples = Y_train.shape[1]
    train_idx = np.random.permutation(n_samples)

    loss1 = 0.
    loss2 = 0.
    batch_i = 1
    for b_i in range(0,n_samples,batch_size):

        optim_d.zero_grad()
        
        batch_idx = train_idx[b_i:(b_i+batch_size)]
        
        cur_batch_size = batch_idx.size

        cur_y = torch.tensor(Y_train[:,batch_idx],device=torch_device,dtype=torch_dtype)
        cur_y = apply_affine(cur_y,torch_device,torch_dtype)
        cur_x = traj2img(cur_y,width,height,torch_dtype,torch_device,rbf_w=None)
        cur_true = torch.tensor(true_class[batch_idx],device=torch_device,dtype=torch.long)

        model_params,pred_class1 = model_d(cur_x) #affine_theta
        dmp_w_net = model_params[:,4:].view(cur_batch_size,n_dmps,n_bfs) 
        
        dmp_learn = DMPLearn(dt, run_time, n_bfs, n_dmps, torch_dtype, dmp_w_net)
        dmp_learn = dmp_learn.to(torch_device)
        
        init_y = torch.cat([t_ones[:cur_batch_size],model_params[:,:4],t_zeros[:cur_batch_size],model_params[:,2:4]],dim=1)
        cur_pred_y = odeint(dmp_learn, init_y, batch_t, method='euler')[:,:,[8,9]]
        
        #final_pred_y = apply_affine(cur_pred_y,cur_batch_size,affine_theta)
        final_pred_y = cur_pred_y
        
        loss_d1 = torch.mean(torch.abs(final_pred_y - cur_y))
        
        loss_d1.backward(retain_graph=True)
        optim_d.step()
        
        
        mnist_data, mnist_target = next(itertools.islice(train_loader, batch_i, None))
        mnist_data, mnist_target = mnist_data.to(torch_device), mnist_target.to(torch_device).long()
        
        #_,pred_class1 = model_d(cur_x)
        loss_c1= F.nll_loss(pred_class1, cur_true)
        
        optim_d.zero_grad()
        
        mnist_model_params,pred_class2 = model_d(mnist_data) #mnist_affine
        loss_c2 = F.nll_loss(pred_class2, mnist_target)
        
        loss_d2 = loss_c2 + loss_c1
        
        loss_d2.backward()
        optim_d.step()
        
        loss1 += loss_d1.item()
        loss2 += loss_d2.item()
        
        if batch_i % print_every_itr == 0:
            print(
            "\t{} ({}) current loss1 = {:.4f}, current loss2 = {:.4f}".format(epoch_idx, batch_i, loss_d1,loss_d2))
            
            print(torch.max(pred_class1,dim=1)[1])
            print(torch.max(pred_class2,dim=1)[1])


            mnist_dmp_w_net = mnist_model_params[:,4:].view(cur_batch_size,n_dmps,n_bfs) 
            
            dmp_learn.set_w(mnist_dmp_w_net)
            
            init_y = torch.cat([t_ones[:cur_batch_size],mnist_model_params[:,:4],t_zeros[:cur_batch_size],mnist_model_params[:,2:4]],dim=1)
            mnist_pred_y = odeint(dmp_learn, init_y, batch_t, method='euler')[:,:,[8,9]]
            
            #final_mnist_pred_y = apply_affine(mnist_pred_y,cur_batch_size,mnist_affine)
            final_mnist_pred_y = mnist_pred_y

            real_img1 = plot_res(cur_pred_y,cur_x,width,height,torch_dtype,torch_device)
            real_img2 = plot_res(final_mnist_pred_y,mnist_data,width,height,torch_dtype,torch_device)
            real_img = torch.cat([real_img1,real_img2])
            
            #real_img = nn.functional.interpolate(real_img, scale_factor=10, mode='bilinear', align_corners=True)
            
            #g_tr = torch.cat([real_img,prod_img],0)
            
            f_save = os.path.join(img_out_dir,'{}_{}.png'.format(epoch_idx, batch_i))
            save_image(real_img,f_save)

        batch_i+=1
        
    print('Epoch {} - loss1 = {:.4f}, loss1 = {:.4f}'.format(epoch_idx,loss1,loss2))
    if epoch_idx % save_every == 0:
        #f_save = os.path.join(model_out_dir,'model_d_epoch_{}.pth'.format(epoch_idx))
        f_save = os.path.join(model_out_dir,'model_d_affine_new.pth')
        torch.save({'state_dict': model_d.state_dict()},f_save)



    

    
    
    
    
    
    
    
    