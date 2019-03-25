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
import math

import sys
sys.path.append(os.path.join('..','imports'))

import pydmps
from torchdiffeq import odeint

from parameters import directory,n_bfs   
from dmp_nn import DMPFunc

def load_trajectories(directory):
    files = os.listdir(directory)
    trajectories = {}
    for name in files:
        filename = os.path.join(directory,name)
        traj = np.load(filename)
        
        if filename[-4:] == '.npz':
            traj = traj['arr_0']
        
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
def traj2img(traj,width,height,torch_dtype,torch_device,rbf_w=500):
    
    yv, xv = torch.meshgrid([torch.linspace(1., 0., width), torch.linspace(0., 1., height)])
    
    is_training = False
    if rbf_w is None:
        rand_rbf_w = torch.randn((1,traj.size(1),1,1),device=torch_device,dtype=torch_dtype)
        #rand_rbf_w.repeat(traj.size(0),1,traj.size(1),xv.size(0),xv.size(1))
        rand_rbf_w = torch.clamp(rand_rbf_w*300 + 600,min=200)
        rbf_w = rand_rbf_w
        is_training = True
    
    
    xv = xv.unsqueeze(0).unsqueeze(0).type(torch_dtype).to(torch_device)
    yv = yv.unsqueeze(0).unsqueeze(0).type(torch_dtype).to(torch_device)
    x_traj = traj[:,:,0].unsqueeze(-1).unsqueeze(-1)
    y_traj = traj[:,:,1].unsqueeze(-1).unsqueeze(-1)
    
    w = torch.max((torch.exp(-((xv-x_traj)**2+(yv-y_traj)**2)*rbf_w)),dim=0)[0]
    
    if is_training == True:
        w[w<0.6] = 0#torch.exp(w)
        w[w!=0] = 1
        
        kernel_ =  np.array([[1,4,1],[4,16,4], [1,4,1]])/16.
        kernel = torch.tensor(kernel_,device=torch_device,dtype=torch_dtype) 
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        #kernel = torch.ones((1,1,3,3),device=torch_device,dtype=torch_dtype)
        
        
        w = F.conv2d(w.unsqueeze(1), kernel,padding=1).squeeze()
        
        #w[w!=0] = torch.exp(w[w!=0])
        #w[w<0.01] = 0
        #w[w>0.2]= 1
        
    return w    


def compute_dmp_params():
    # Computing the DMP parameters
    trajectories = load_trajectories(directory)
    all_keys = sorted(trajectories.keys())
    y_track_dmp = {}
    dmp_w = {}
    orig_y = {}
    for i_k,key in enumerate(all_keys):
        y_track_dmp[key] = []
        dmp_w[key] = []
        orig_y[key] = []
        for stroke_k in range(len(trajectories[key])):
            y_des = np.array(trajectories[key][stroke_k])  
                    
            n_dmps= y_des.shape[1] 
            dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=n_dmps, n_bfs=n_bfs, ay=np.ones(2)*10.0)
            dmp.imitate_path(y_des=y_des.T, plot=False)
                
            y_track, _, _ = dmp.rollout()
            y_track_dmp[key].append(y_track)
            dmp_w[key].append(dmp.w)
            orig_y[key].append(y_des)
            
    return dmp_w, y_track_dmp, orig_y        



def generate_trajectories_from_template(batch_size_per_digit,n_bfs,dt,run_time,torch_device,torch_dtype):
     
    #plot_trajectories(trajectories, y_track_dmp, plot_dmp=plot_dmp)
    
    
    # Generate dataset
    # Create a batch of initial states and DMP parameters
    #batch_size_per_digit = 2560
    batch_y0 = []
    batch_w = []
    batch_rand = []
    rand_factor = [1000,100,50,100,20,50,50,50,1000,50]
    true_class = []
    for i_k,key in enumerate(all_keys):
        canonical_x = np.array([1.,1.])
        traj_goal = trajectories[key][0][-1,:]
        traj_start = trajectories[key][0][0,:]
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



def plot_multi_stroke(saved_traj,title=None):        
    colors = plt.cm.get_cmap('hsv', len(saved_traj)+1)
    for c_i,cur_stroke in enumerate(saved_traj):
        cs = np.array(cur_stroke)
        plt.plot(cs[:,0],cs[:,1],c=colors(c_i))
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if title is not None:
        plt.title(title)        
    plt.axis('off')  
    
    
    
def affine_transform(pred_y,torch_device,torch_dtype):
    pred_y = pred_y * (torch.rand((1,pred_y.size(1),2),device=torch_device,dtype=torch_dtype)+1) 
    max_traj = (torch.max(pred_y,0))[0]
    min_traj = (torch.min(pred_y,0))[0]
    traj_div = torch.max(max_traj-min_traj,dim=1)[0]/1.3#1.8 
    traj_offset = min_traj+(max_traj-min_traj)/2. + (torch.rand((pred_y.size(1),2),device=torch_device,dtype=torch_dtype)-.5)/5
    pred_y = (pred_y-traj_offset)/traj_div.unsqueeze(-1).repeat(1,traj_offset.size(1))/2.+0.5
    return pred_y    




def plot_res(cur_pred_y,cur_x,width,height,torch_dtype,torch_device):
    plt_samples = 80#64
    prod_img = traj2img(cur_pred_y[:,:plt_samples,:],width,height,torch_dtype,torch_device,rbf_w=2000)
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

def apply_affine(cur_pred_y,torch_device,torch_dtype):
    #tot_samp = cur_pred_y.size(1)*cur_pred_y.size(0)
    #tmp_pred_y = torch.cat([cur_pred_y.view(-1,2),cat_ones[:tot_samp]],dim=1).unsqueeze(-1)
    
    rotation_m = torch.zeros((cur_pred_y.size(1),2,2),device=torch_device,dtype=torch_dtype)
    angles = (torch.rand(cur_pred_y.size(1),device=torch_device,dtype=torch_dtype)-0.5)*2*math.pi/6
    rotation_m[:,0,0] = torch.cos(angles)
    rotation_m[:,0,1] = torch.sin(angles)
    rotation_m[:,1,0] = -torch.sin(angles)
    rotation_m[:,1,1] = torch.cos(angles)
    
    shear_m = torch.ones((cur_pred_y.size(1),2,2),device=torch_device,dtype=torch_dtype)
    shear_f = (torch.rand(cur_pred_y.size(1),device=torch_device,dtype=torch_dtype)-0.5)*1
    shear_m[:,1,0] = 0
    shear_m[:,0,1] = shear_f
    
    affine_theta = torch.bmm(shear_m,rotation_m)
    
    tmp_pred_y = cur_pred_y.view(-1,2).unsqueeze(-1)
    m1 = affine_theta.repeat(cur_pred_y.size(0),1,1)
    m2 = tmp_pred_y
    final_pred_y = torch.bmm(m1,m2).view(cur_pred_y.size())
    
    final_pred_y = affine_transform(final_pred_y,torch_device,torch_dtype)
    
    return final_pred_y


