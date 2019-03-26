#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:52:42 2019

@author: giuseppelisi
"""
import os
import numpy as np

from utils import generate_trajectories_from_template,plot_res,apply_affine,traj2img,compute_dmp_params,plot_multi_stroke

import torch
from dmp_nn import DMPLearn, Encoder,DMPFunc
from torch import nn, optim

from torchvision.utils import save_image

from torchdiffeq import odeint_adjoint 
from torchdiffeq import odeint

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import itertools

import torch.nn.functional as F

from parameters import n_bfs,dt,run_time,torch_device,torch_dtype,model_out_dir,img_out_dir,n_dmps,width,height

import matplotlib.pyplot as plt

#torch.set_printoptions(precision=10)

batch_size = 256#64#256
epochs = 250

# Compute the parameters of the prototypes
dmp_w, y_track_dmp, orig_y  = compute_dmp_params()

# Define all the models
all_models = {}
all_optim = {}
bias = {}
for i_k,key in enumerate(sorted(dmp_w.keys())):
    cur_model = Encoder(n_bfs*n_dmps,dmp_w[key],orig_y[key],torch_device,torch_dtype)
    cur_model.to(torch_device).type(torch_dtype)
    cur_optim = torch.optim.Adadelta(cur_model.parameters())#optim.SGD(cur_model.parameters(), lr=0.01)
    all_models[key] = cur_model
    all_optim[key] = cur_optim
    

for cur_model_i,cur_key in enumerate(all_models.keys()):    
    dir_save = os.path.join(img_out_dir,cur_key)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)   
                                

t_zeros = torch.zeros(batch_size,n_dmps,device=torch_device,dtype=torch_dtype)
t_ones = torch.ones(batch_size,n_dmps,device=torch_device,dtype=torch_dtype)
batch_t = torch.arange(0.,run_time,dt, device=torch_device,dtype=torch_dtype)

print_every_itr = 25
save_every = 1

train_dataset = datasets.MNIST(root='../data',
            train=True,
            download=False,
            transform=transforms.ToTensor())
    
train_loader = DataLoader(train_dataset, shuffle=True,
        batch_size=batch_size)


dmp_learn = DMPLearn(dt, run_time, n_bfs, n_dmps, torch_dtype)
dmp_learn = dmp_learn.to(torch_device)

rand_factor = [1000,100,100,50,100,20,50,50,50,50,1000,50]

#model_d.train()
for epoch_idx in range(epochs):
    
    for batch_idx, (mnist_x, mnist_y) in enumerate(train_loader):
    
        # for each digit load a different model
        for i_num in range(10):
            
            cur_samples = mnist_y == i_num
            cur_x = mnist_x[cur_samples].to(torch_device).type(torch_dtype)
            cur_y = mnist_y[cur_samples].to(torch_device).type(torch_dtype)
            cur_batch_size = cur_x.size(0)
            
            all_losses = np.zeros(len(all_models))
            
            # loop across all models and use only those associated witha class
            for cur_model_i,cur_key in enumerate(all_models.keys()):
                
                if str(i_num) in cur_key:
                    #print(cur_key)
                    
                    cur_model = all_models[cur_key]
                    cur_optim = all_optim[cur_key]
                    all_model_params = cur_model(cur_x)
                    
                    proto_traj = orig_y[cur_key]
                    
                    with torch.no_grad():
                        #plt.figure()
                        synth_traj = []
                        for i_m,model_params in enumerate(all_model_params):
                            
                            #model_params += torch.randn_like(model_params)*0.001
                            
                            end_t = torch.tensor(proto_traj[i_m][-1],device=torch_device,dtype=torch_dtype).view(1,-1).repeat(cur_batch_size,1)
                            start_t = torch.tensor(proto_traj[i_m][0],device=torch_device,dtype=torch_dtype).view(1,-1).repeat(cur_batch_size,1)
                            
                            dmp_w_net = model_params.view(cur_batch_size,n_dmps,n_bfs) 
                            dmp_w_net += torch.randn_like(dmp_w_net)*rand_factor[cur_model_i]#(torch.max(torch.abs(dmp_w_net))*0.5)
                            
                            dmp_learn.set_w(dmp_w_net)
                            
                            init_y = torch.cat([t_ones[:cur_batch_size],end_t,start_t,t_zeros[:cur_batch_size],start_t],dim=1)
                            cur_pred_y = odeint(dmp_learn, init_y, batch_t, method='euler')[:,:,[8,9]]
                            synth_traj.append(cur_pred_y)
                            
#                            plt_pred = cur_pred_y.cpu().numpy()
#                            plot_multi_stroke(y_track_dmp[cur_key])
#                            plt.plot(plt_pred[:,0,0],plt_pred[:,0,1],'k')
#                            plt.xlim([0, 1])
#                            plt.ylim([0, 1])
                        
                        traj_combo = torch.cat(synth_traj,dim=0)  
                        traj_combo = apply_affine(traj_combo,torch_device,torch_dtype)
                        
                        traj_img = traj2img(traj_combo,width,height,torch_dtype,torch_device,rbf_w=None)
                        
#                        plt.figure()
#                        plt.imshow(traj_img[0].cpu().numpy())
#                        plt.show()
                        
                    #plt.show()
                        
                    cur_optim.zero_grad()
                    all_model_params = cur_model(traj_img)
                    
                    loss = 0.
                    pred_y = []
                    for i_m,model_params in enumerate(all_model_params):
                        
                        dmp_w_net = model_params.view(cur_batch_size,n_dmps,n_bfs) 
                        dmp_learn.set_w(dmp_w_net)
                        
                        end_t = torch.tensor(proto_traj[i_m][-1],device=torch_device,dtype=torch_dtype).view(1,-1).repeat(cur_batch_size,1)
                        start_t = torch.tensor(proto_traj[i_m][0],device=torch_device,dtype=torch_dtype).view(1,-1).repeat(cur_batch_size,1)
                        
                        init_y = torch.cat([t_ones[:cur_batch_size],end_t,start_t,t_zeros[:cur_batch_size],start_t],dim=1)
                        cur_pred_y = odeint(dmp_learn, init_y, batch_t, method='euler')[:,:,[8,9]]
                        pred_y.append(cur_pred_y)
                        
                    pred_y = torch.cat(pred_y,dim=0)   
                    cur_loss = torch.mean(torch.abs(pred_y - traj_combo))
                    loss+=cur_loss
                        
                    
                    loss.backward()
                    cur_optim.step()
                    
                    all_losses[cur_model_i] += loss.item()
                    
                      
                    
                    if batch_idx % print_every_itr==0:
                        
                        print ('{} ({}, {}) current loss1 = {:.4f}'.format(epoch_idx,batch_idx,cur_key,loss.item()))
                        
                        with torch.no_grad():
                            
                             
                            
                            all_model_params = cur_model(cur_x)
                            pred_traj = []
                            for i_m,model_params in enumerate(all_model_params):
                                dmp_w_net = model_params.view(cur_batch_size,n_dmps,n_bfs) 
                                dmp_learn.set_w(dmp_w_net)
                                
                                end_t = torch.tensor(proto_traj[i_m][-1],device=torch_device,dtype=torch_dtype).view(1,-1).repeat(cur_batch_size,1)
                                start_t = torch.tensor(proto_traj[i_m][0],device=torch_device,dtype=torch_dtype).view(1,-1).repeat(cur_batch_size,1)
                                
                                init_y = torch.cat([t_ones[:cur_batch_size],end_t,start_t,t_zeros[:cur_batch_size],start_t],dim=1)
                                cur_pred_y = odeint(dmp_learn, init_y, batch_t, method='euler')[:,:,[8,9]]
                                pred_traj.append(cur_pred_y)
                                
                            pred_traj_combo = torch.cat(pred_traj,dim=0)      
                            real_img1 = plot_res(pred_traj_combo,cur_x,width,height,torch_dtype,torch_device)
                            real_img2 = plot_res(pred_y,traj_img,width,height,torch_dtype,torch_device)
                            real_img = torch.cat([real_img1,real_img2])
                            f_save = os.path.join(img_out_dir,cur_key,'{}_{}.png'.format(epoch_idx, batch_idx))
                            save_image(real_img,f_save)
                   
                    
    print('Epoch {} - loss = {}'.format(epoch_idx,all_losses))
    if epoch_idx % save_every == 0:
        for cur_model_i,cur_key in enumerate(all_models.keys()):
            f_save = os.path.join(model_out_dir,'model_{}.pth'.format(cur_key))
            torch.save({'state_dict': all_models[cur_key].state_dict()},f_save)    
                        
                    
                    
                    
                    
            
            
            
            
        
        
        
        
        
        
        
        
#        optim_d.zero_grad()
#        
#        batch_idx = train_idx[b_i:(b_i+batch_size)]
#        
#        cur_batch_size = batch_idx.size
#
#        cur_y = torch.tensor(Y_train[:,batch_idx],device=torch_device,dtype=torch_dtype)
#        cur_y = apply_affine(cur_y,torch_device,torch_dtype)
#        cur_x = traj2img(cur_y,width,height,torch_dtype,torch_device,rbf_w=None)
#        cur_true = torch.tensor(true_class[batch_idx],device=torch_device,dtype=torch.long)
#
#        model_params,pred_class1 = model_d(cur_x) #affine_theta
#        dmp_w_net = model_params[:,4:].view(cur_batch_size,n_dmps,n_bfs) 
#        
#        dmp_learn = DMPLearn(dt, run_time, n_bfs, n_dmps, torch_dtype, dmp_w_net)
#        dmp_learn = dmp_learn.to(torch_device)
#        
#        init_y = torch.cat([t_ones[:cur_batch_size],model_params[:,:4],t_zeros[:cur_batch_size],model_params[:,2:4]],dim=1)
#        cur_pred_y = odeint(dmp_learn, init_y, batch_t, method='euler')[:,:,[8,9]]
#        
#        #final_pred_y = apply_affine(cur_pred_y,cur_batch_size,affine_theta)
#        final_pred_y = cur_pred_y
#        
#        loss_d1 = torch.mean(torch.abs(final_pred_y - cur_y))
#        
#        loss_d1.backward(retain_graph=True)
#        optim_d.step()
#        
#        
#        mnist_data, mnist_target = next(itertools.islice(train_loader, batch_i, None))
#        mnist_data, mnist_target = mnist_data.to(torch_device), mnist_target.to(torch_device).long()
#        
#        #_,pred_class1 = model_d(cur_x)
#        loss_c1= F.nll_loss(pred_class1, cur_true)
#        
#        optim_d.zero_grad()
#        
#        mnist_model_params,pred_class2 = model_d(mnist_data) #mnist_affine
#        loss_c2 = F.nll_loss(pred_class2, mnist_target)
#        
#        loss_d2 = loss_c2 + loss_c1
#        
#        loss_d2.backward()
#        optim_d.step()
#        
#        loss1 += loss_d1.item()
#        loss2 += loss_d2.item()
#        
#        if batch_i % print_every_itr == 0:
#            print(
#            "\t{} ({}) current loss1 = {:.4f}, current loss2 = {:.4f}".format(epoch_idx, batch_i, loss_d1,loss_d2))
#            
#            print(torch.max(pred_class1,dim=1)[1])
#            print(torch.max(pred_class2,dim=1)[1])
#
#
#            mnist_dmp_w_net = mnist_model_params[:,4:].view(cur_batch_size,n_dmps,n_bfs) 
#            
#            dmp_learn.set_w(mnist_dmp_w_net)
#            
#            init_y = torch.cat([t_ones[:cur_batch_size],mnist_model_params[:,:4],t_zeros[:cur_batch_size],mnist_model_params[:,2:4]],dim=1)
#            mnist_pred_y = odeint(dmp_learn, init_y, batch_t, method='euler')[:,:,[8,9]]
#            
#            #final_mnist_pred_y = apply_affine(mnist_pred_y,cur_batch_size,mnist_affine)
#            final_mnist_pred_y = mnist_pred_y
#
#            real_img1 = plot_res(cur_pred_y,cur_x,width,height,torch_dtype,torch_device)
#            real_img2 = plot_res(final_mnist_pred_y,mnist_data,width,height,torch_dtype,torch_device)
#            real_img = torch.cat([real_img1,real_img2])
#            
#            #real_img = nn.functional.interpolate(real_img, scale_factor=10, mode='bilinear', align_corners=True)
#            
#            #g_tr = torch.cat([real_img,prod_img],0)
#            
#            f_save = os.path.join(img_out_dir,'{}_{}.png'.format(epoch_idx, batch_i))
#            save_image(real_img,f_save)
#
#        batch_i+=1
#        
#    print('Epoch {} - loss1 = {:.4f}, loss1 = {:.4f}'.format(epoch_idx,loss1,loss2))
#    if epoch_idx % save_every == 0:
#        #f_save = os.path.join(model_out_dir,'model_d_epoch_{}.pth'.format(epoch_idx))
#        f_save = os.path.join(model_out_dir,'model_d_affine_new.pth')
#        torch.save({'state_dict': model_d.state_dict()},f_save)
#
#
#
#    

    
    
    
    
    
    
    
    