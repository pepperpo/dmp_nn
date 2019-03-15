#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:46:58 2019

@author: giuseppelisi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.join('..','imports'))

import pydmps
from torchdiffeq import odeint

from parameters import directory   
from dmp_nn import DMPFunc

def load_trajectories(directory):
    files = os.listdir(directory)
    trajectories = {}
    for name in files:
        filename = os.path.join(directory,name)
        traj = np.load(filename)['arr_0']
        trajectories[name[:-4]] = traj
    return trajectories    



def plot_trajectories(trajectories, y_track_dmp, plot_dmp=True):
    if plot_dmp == True:
        all_keys = sorted(trajectories.keys())
        n_traj = len(trajectories)
        # Plotting    
        plt.figure(figsize=(2,12))    
        for i_k,key in enumerate(all_keys):    
        
            y_des = trajectories[key]    
            y_track = y_track_dmp[key]
            
            plt.subplot(n_traj,2,i_k*2+1)
            plt.plot(y_des[:,0],y_des[:,1])
            plt.axis('equal')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.axis('off')
            
            plt.subplot(n_traj,2,i_k*2+2)
            plt.plot(y_track[:,0],y_track[:,1])
            plt.axis('equal')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.axis('off')
    
        plt.show()
        
        
def plot_ode_output(pred_y,img,all_keys,batch_size_per_digit,plot_ode_out=True,plot_num = 10):
    if plot_ode_out:
        plt.figure(figsize=(14,14))
        n_traj = len(all_keys)
        for i_k in range(n_traj):
            cur_idx = np.arange(i_k*batch_size_per_digit,i_k*batch_size_per_digit+batch_size_per_digit)
            rand_choice = np.random.choice(batch_size_per_digit, plot_num)
            pick_idx = cur_idx[rand_choice]
            for i_p,pick in enumerate(pick_idx):
                spl_i = np.ravel_multi_index([i_k,i_p*2], [n_traj, plot_num*2], order='C')
                plt.subplot(n_traj,plot_num*2,spl_i+1)
                plt.plot(pred_y[:,pick,0],pred_y[:,pick,1])
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.axis('off')
                
                spl_i = np.ravel_multi_index([i_k,i_p*2+1], [n_traj, plot_num*2], order='C')
                plt.subplot(n_traj,plot_num*2,spl_i+1)
                plt.imshow(img[pick],cmap='gray')
                plt.axis('off')
                
        plt.show()        
        
        
# traj [Time,Trial,Dimensions]
def traj2img_old(traj,torch_device,torch_dtype,rbf_w=500):
    
    width=28
    height =28
    
    yv, xv = torch.meshgrid([torch.linspace(1., 0., width), torch.linspace(0., 1., height)])
    
    if rbf_w is None:
        rand_rbf_w = torch.rand((1,traj.size(1),1,1),device=torch_device,dtype=torch_dtype)
        #rand_rbf_w.repeat(traj.size(0),1,traj.size(1),xv.size(0),xv.size(1))
        rand_rbf_w = rand_rbf_w*200 + 50
        rbf_w = rand_rbf_w
    
    
    xv = xv.unsqueeze(0).unsqueeze(0).type(torch_dtype).to(torch_device)
    yv = yv.unsqueeze(0).unsqueeze(0).type(torch_dtype).to(torch_device)
    x_traj = traj[:,:,0].unsqueeze(-1).unsqueeze(-1)
    y_traj = traj[:,:,1].unsqueeze(-1).unsqueeze(-1)
    
    w = torch.max((torch.exp(-((xv-x_traj)**2+(yv-y_traj)**2)*rbf_w)),dim=0)[0]
        
    return w   










def traj2img_(traj_in,img,prev_w,n_samp,size_big,torch_device,torch_dtype):
    
    if prev_w!=0:
        traj = (1-prev_w)*traj_in[1:] + prev_w*traj_in[:-1]    
        separation = torch.sqrt((traj_in[1:,:,0]-traj_in[:-1,:,0])**2 + (traj_in[1:,:,1]-traj_in[:-1,:,1])**2)
        #intense = 2*torch.clamp(separation-1,min=0,max=1).flatten()
        min_t = torch.min(separation,dim=0)[0]
        max_t = torch.max(separation,dim=0)[0]
        intense = 2*((separation-min_t)/(max_t-min_t)).flatten()
    else:
        intense = 2
        traj = traj_in
    
    time_samp = traj.size(0)
    
    # convert to pixels
    t_lo = torch.floor(traj)
    t_hi = torch.ceil(traj)
    t_hi[t_hi==t_lo] = t_hi[t_hi==t_lo]+1
    
    #distance between continuous trajectory and discrete trajectory
    wtlo = (t_hi-traj).view(-1,2)
    wthi = (traj-t_lo).view(-1,2)
    
    # clamping within the image margins
    t_lo = torch.clamp(t_lo, min=0, max=size_big-1).long().view(-1,2)
    t_hi = torch.clamp(t_hi, min=0, max=size_big-1).long().view(-1,2)

    samp_idx = torch.arange(n_samp).unsqueeze(0).repeat(time_samp,1).flatten()
    
    # bilinear interpolation
    loloinc = wtlo[:,1] * wtlo[:,0]
    hiloinc = wthi[:,1] * wtlo[:,0]
    lohiinc = wtlo[:,1] * wthi[:,0]
    hihiinc = wthi[:,1] * wthi[:,0]
    
    img[samp_idx,t_lo[:,1],t_lo[:,0]] = img[samp_idx,t_lo[:,1],t_lo[:,0]] + intense*loloinc
    img[samp_idx,t_hi[:,1],t_lo[:,0]] = img[samp_idx,t_hi[:,1],t_lo[:,0]] + intense*hiloinc
    img[samp_idx,t_lo[:,1],t_hi[:,0]] = img[samp_idx,t_lo[:,1],t_hi[:,0]] + intense*lohiinc
    img[samp_idx,t_hi[:,1],t_hi[:,0]] = img[samp_idx,t_hi[:,1],t_hi[:,0]] + intense*hihiinc
    
    return img



# http://www.cs.toronto.edu/~hinton/code/drawtraj.m
#traj [Time,Trial,Dimensions]
def traj2img(traj,torch_device,torch_dtype,rbf_w=500):
    
    size_big = 36
    traj = traj*26+5
    traj[:,:,1] = size_big-traj[:,:,1]
    
    n_samp= traj.size(1)
    img = torch.zeros([n_samp,size_big,size_big],device=torch_device,dtype=torch_dtype)
    
    img = traj2img_(traj,img,0.,n_samp,size_big,torch_device,torch_dtype)
    img = traj2img_(traj,img,0.75,n_samp,size_big,torch_device,torch_dtype)
    img = traj2img_(traj,img,0.5,n_samp,size_big,torch_device,torch_dtype)
    img = traj2img_(traj,img,0.25,n_samp,size_big,torch_device,torch_dtype)

    #kernel = torch.ones((1,1,3,3),device=torch_device,dtype=torch_dtype)
    
    
    th = 0.5/2
    ink = 20
    kernel = 1.5 * ink * (1+th)* torch.tensor([[th/12, th/6, th/12], [th/6, 1-th, th/6], [th/12, th/6, th/12]],device=torch_device,dtype=torch_dtype)
    
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    conv_img = F.conv2d(img.unsqueeze(1), kernel)
    conv_img = F.conv2d(conv_img, kernel)
    conv_img = F.conv2d(conv_img, kernel)
    conv_img = F.conv2d(conv_img, kernel).squeeze()
    
    print(conv_img.size())
    
    return conv_img   

         






def generate_trajectories_from_template(batch_size_per_digit,n_bfs,dt,run_time,torch_device,torch_dtype):
    
    # Computing the DMP parameters
    trajectories = load_trajectories(directory)
    all_keys = sorted(trajectories.keys())
    
    y_track_dmp = {}
    dmp_w = {}
    for i_k,key in enumerate(all_keys):
        y_des = trajectories[key]    
        n_dmps= y_des.shape[1] 
        dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=n_dmps, n_bfs=n_bfs, ay=np.ones(2)*10.0)
        dmp.imitate_path(y_des=y_des.T, plot=False)
            
        y_track, _, _ = dmp.rollout()
        y_track_dmp[key] = y_track
        dmp_w[key] = dmp.w
        
        
     
    #plot_trajectories(trajectories, y_track_dmp, plot_dmp=plot_dmp)
    
    
    # Generate dataset
    # Create a batch of initial states and DMP parameters
    #batch_size_per_digit = 2560
    batch_y0 = []
    batch_w = []
    batch_rand = []
    rand_factor = [1000,100,50,50,50,50,50,50,1000,50]
    true_class = []
    for i_k,key in enumerate(all_keys):
        canonical_x = np.array([1.,1.])
        traj_goal = trajectories[key][-1,:]
        traj_start = trajectories[key][0,:]
        d_traj = np.array([0.,0.])
        cur_y0 = torch.tensor(np.concatenate([canonical_x,traj_goal,traj_start,d_traj,traj_start]),device=torch_device,dtype=torch_dtype).unsqueeze(0).repeat(batch_size_per_digit,1)
        
        cur_w = torch.tensor(dmp_w[key], device=torch_device,dtype=torch_dtype)
        cur_w = cur_w.unsqueeze(0).repeat(batch_size_per_digit,1,1)
        
        batch_rand.append(torch.randn_like(cur_w)*rand_factor[i_k])
        batch_y0.append(cur_y0)
        batch_w.append(cur_w)
        true_class.extend([i_k]*batch_size_per_digit)
    
    batch_y0 = torch.cat(batch_y0)
    batch_w = torch.cat(batch_w)
    batch_rand = torch.cat(batch_rand)
    
    batch_w = batch_w + batch_rand
    
    print('The size of the batch W is {}'.format(batch_w.size()))
    
    func = DMPFunc(dt, run_time, n_bfs, n_dmps, torch_dtype, init_w=batch_w)
    func = func.to(torch_device)
    
    batch_t = torch.arange(0.,run_time,dt, device=torch_device,dtype=torch_dtype)
    
    import time
    t0 = time.time()
    pred_y = odeint(func, batch_y0, batch_t,method='euler')
    print('odeint took {} s'.format(time.time()-t0))
    
    pred_y = pred_y[:,:,[8,9]]
    
    true_class = np.array(true_class)
#    from sklearn.preprocessing import OneHotEncoder
#    onehot_encoder = OneHotEncoder(sparse=False)
#    onehot_encoded = onehot_encoder.fit_transform(np.array(true_class)[:,None])
    
    return pred_y,batch_t,true_class














