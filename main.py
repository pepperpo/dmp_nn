#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:52:42 2019

@author: giuseppelisi
"""
import os
import numpy as np

from utils import load_trajectories, plot_trajectories, plot_ode_output,generate_trajectories_from_template

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

# Pythorch variables
if torch.cuda.is_available():
    torch_device = torch.device('cuda:0')
    using_cuda = True
else:
    torch_device = torch.device('cpu')   
    using_cuda = False
torch_dtype = torch.float32    
    
# Plotting variables
plot_dmp = False
plot_ode_out = False

# DMP input
n_bfs = 50
n_dmps = 2
dt = .01
run_time = 1.

# Image generation
width = 28
height = 28


train_out_dir = 'train_out'
img_out_dir = os.path.join(train_out_dir,'img_out')
model_out_dir = os.path.join(train_out_dir,'model_out')

if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)
    
if not os.path.exists(model_out_dir):
    os.makedirs(model_out_dir)    


# traj [Time,Trial,Dimensions]
def traj2img(traj,rbf_w=500):
    
    yv, xv = torch.meshgrid([torch.linspace(1., 0., width), torch.linspace(0., 1., height)])
    
    if rbf_w is None:
        rand_rbf_w = torch.rand((1,traj.size(1),1,1),device=torch_device,dtype=torch_dtype)
        #rand_rbf_w.repeat(traj.size(0),1,traj.size(1),xv.size(0),xv.size(1))
        rand_rbf_w = rand_rbf_w*600 + 400
        rbf_w = rand_rbf_w
    
    
    xv = xv.unsqueeze(0).unsqueeze(0).type(torch_dtype).to(torch_device)
    yv = yv.unsqueeze(0).unsqueeze(0).type(torch_dtype).to(torch_device)
    x_traj = traj[:,:,0].unsqueeze(-1).unsqueeze(-1)
    y_traj = traj[:,:,1].unsqueeze(-1).unsqueeze(-1)
    
    w = torch.max((torch.exp(-((xv-x_traj)**2+(yv-y_traj)**2)*rbf_w)),dim=0)[0]
    w[w<0.1] = 0
    w[w>0.2]= 1
        
    return w    
    

def affine_transform(pred_y):
    pred_y = pred_y * (torch.rand((1,pred_y.size(1),2),device=torch_device,dtype=torch_dtype)+1) 
    max_traj = (torch.max(pred_y,0))[0]
    min_traj = (torch.min(pred_y,0))[0]
    traj_div = torch.max(max_traj-min_traj,dim=1)[0]/1.5#1.8 
    traj_offset = min_traj+(max_traj-min_traj)/2.
    pred_y = (pred_y-traj_offset)/traj_div.unsqueeze(-1).repeat(1,traj_offset.size(1))/2.+0.5
    return pred_y
    


def plot_res(cur_pred_y,cur_x):
    plt_samples = 80#64
    prod_img = traj2img(cur_pred_y[:,:plt_samples,:],rbf_w=2000)
    real_img = cur_x[:plt_samples]
    
    real_img = real_img.data.view(
        plt_samples, 1, 28,28).cpu().repeat(1,3,1,1)
    
    prod_img = prod_img.data.view(
        plt_samples, 28,28).cpu()
    
    img_mask = prod_img>.001
    
    ch0 = real_img[:,0,:,:]
    ch1 = real_img[:,1,:,:]
    ch2 = real_img[:,2,:,:]
    
    ch0[img_mask] = (1-prod_img[img_mask])*ch0[img_mask]
    ch1[img_mask] = (1-prod_img[img_mask])*ch1[img_mask]
    ch2[img_mask] = prod_img[img_mask] + (1-prod_img[img_mask])*ch2[img_mask]
    
    real_img[:,0,:,:] = ch0
    real_img[:,1,:,:] = ch1
    real_img[:,2,:,:] = ch2
    
    return real_img


def apply_affine(cur_pred_y,cur_batch_size,affine_theta):
    tot_samp = cur_batch_size*cur_pred_y.size(0)
    tmp_pred_y = torch.cat([cur_pred_y.view(-1,2),cat_ones[:tot_samp]],dim=1).unsqueeze(-1)
    m1 = affine_theta.repeat(cur_pred_y.size(0),1,1)
    m2 = tmp_pred_y
    final_pred_y = torch.bmm(m1,m2).view(cur_pred_y.size())
    return final_pred_y


pred_y,batch_t,true_class = generate_trajectories_from_template(2560,n_bfs,dt,run_time,torch_device,torch_dtype)

#pred_y = affine_transform(pred_y)
#Y_train = pred_y.cpu().numpy()
#X_plot = traj2img(pred_y,rbf_w=None)
#X_plot = X_plot.cpu().numpy()
#plot_ode_output(Y_train,X_plot,all_keys,batch_size_per_digit,plot_ode_out=plot_ode_out,plot_num = 10)


Y_train = pred_y.cpu().numpy()


if using_cuda:
    print('releasing GPU cache of dataset generator')
    del pred_y
    torch.cuda.empty_cache()


n_samples = Y_train.shape[1]
batch_size = 256#64#256
epochs = 250
train_idx = np.random.permutation(n_samples)

model_d = ModelD(n_bfs*n_dmps+n_dmps*2)
model_d.to(torch_device)
optim_d = optim.SGD(model_d.parameters(), lr=0.01)

t_zeros = torch.zeros(batch_size,n_dmps,device=torch_device,dtype=torch_dtype)
t_ones = torch.ones(batch_size,n_dmps,device=torch_device,dtype=torch_dtype)

print_every_itr = 25
save_every = 1


train_dataset = datasets.MNIST(root='data',
            train=True,
            download=False,
            transform=transforms.ToTensor())
    
train_loader = DataLoader(train_dataset, shuffle=True,
        batch_size=batch_size)



cat_ones = torch.ones((101*batch_size,1),device=torch_device,dtype=torch_dtype)


model_d.train()
for epoch_idx in range(epochs):

    loss1 = 0.
    loss2 = 0.
    batch_i = 1
    for b_i in range(0,n_samples,batch_size):

        optim_d.zero_grad()
        
        batch_idx = train_idx[b_i:(b_i+batch_size)]
        
        cur_batch_size = batch_idx.size

        cur_y = torch.tensor(Y_train[:,batch_idx],device=torch_device,dtype=torch_dtype)
        cur_y = affine_transform(cur_y)
        cur_x = traj2img(cur_y,rbf_w=None)
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

            real_img1 = plot_res(cur_pred_y,cur_x)
            real_img2 = plot_res(final_mnist_pred_y,mnist_data)
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



    

    
    
    
    
    
    
    
    