#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:43:55 2019

@author: giuseppelisi
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DMPFunc(nn.Module):

    def __init__(self, dt, run_time, n_bfs, n_dmp, dtype, init_w=None):
        super(DMPFunc, self).__init__()
        
        self.register_buffer('error_coupling', torch.tensor([1.],dtype=dtype))
        self.register_buffer('ax', torch.tensor([1.],dtype=dtype))
        self.register_buffer('tau', torch.tensor([1.],dtype=dtype))
        self.register_buffer('dt', torch.tensor([dt],dtype=dtype))
        
        self.n_dmp = n_dmp
        
        ### Define the centers
        # desired activations throughout time
        des_c = np.linspace(0, run_time, n_bfs)
        
        c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            c[n] = np.exp(-self.ax.item() * des_c[n])
            
        self.register_buffer('c',torch.tensor(c,dtype=dtype))    
  
        # Define h
        h = np.ones(n_bfs) * n_bfs**1.5 / c / self.ax.item()
        self.register_buffer('h',torch.tensor(h,dtype=dtype)) 
        
        
        self.register_buffer('ay', torch.tensor([10.]*n_dmp,dtype=dtype))
        self.register_buffer('by', torch.tensor([10./4.]*n_dmp,dtype=dtype))
                
        # TODO check_offset
        
#        self.net = nn.Sequential(
#            nn.Linear(n_bfs, n_dmp),
#        )
        
        self.register_buffer('w', init_w)


    def forward(self, t, y):
        
        y = y.view(y.size(0),-1,self.n_dmp) 
        
        x = y[:,0,:]
        goal = y[:,1,:]
        y0 = y[:,2,:]
        d_traj = y[:,3,:]
        traj= y[:,4,:]
        
        delta_x = (-self.ax * x * self.error_coupling) * self.tau 
        
        psi = torch.exp(-self.h * (x[0,0] - self.c)**2)
        
        front_term = x * (goal - y0)
        
        psi_sum = torch.sum(psi)
        
        psi = psi.unsqueeze(0).unsqueeze(-1).repeat(y.size(0),1,1)
        
        dot_prod = torch.bmm(self.w,psi).squeeze()
        
        f = (front_term*dot_prod)/psi_sum
        
        dd_traj = (self.ay *
                           (self.by * (goal - traj) -
                           d_traj/self.tau) + f) * self.tau
                   
        delta_d_traj = dd_traj * self.tau * self.error_coupling
        new_d_traj = d_traj + delta_d_traj*self.dt  
        delta_traj = new_d_traj * self.error_coupling
             
        ret_tensor = torch.cat([delta_x,torch.zeros_like(goal),torch.zeros_like(y0),delta_d_traj,delta_traj],dim=1)
        
        return ret_tensor   
    
    
    
    
    
    
    
    
    
    
    
    
    
class DMPLearn(nn.Module):

    def __init__(self, dt, run_time, n_bfs, n_dmp, dtype, init_w):
        super(DMPLearn, self).__init__()
        
        self.register_buffer('error_coupling', torch.tensor([1.],dtype=dtype))
        self.register_buffer('ax', torch.tensor([1.],dtype=dtype))
        self.register_buffer('tau', torch.tensor([1.],dtype=dtype))
        self.register_buffer('dt', torch.tensor([dt],dtype=dtype))
        
        self.n_dmp = n_dmp
        
        ### Define the centers
        # desired activations throughout time
        des_c = np.linspace(0, run_time, n_bfs)
        
        c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            c[n] = np.exp(-self.ax.item() * des_c[n])
            
        self.register_buffer('c',torch.tensor(c,dtype=dtype))    
  
        # Define h
        h = np.ones(n_bfs) * n_bfs**1.5 / c / self.ax.item()
        self.register_buffer('h',torch.tensor(h,dtype=dtype)) 
        
        
        self.register_buffer('ay', torch.tensor([10.]*n_dmp,dtype=dtype))
        self.register_buffer('by', torch.tensor([10./4.]*n_dmp,dtype=dtype))
                
        # TODO check_offset
        
#        self.net = nn.Sequential(
#            nn.Linear(n_bfs, n_dmp),
#        )
        
        self.w = init_w


    def set_w(self,w):
        self.w=w


    def forward(self, t, y):
        
        y = y.view(y.size(0),-1,self.n_dmp) 
        
        x = y[:,0,:]
        goal = y[:,1,:]
        y0 = y[:,2,:]
        d_traj = y[:,3,:]
        traj= y[:,4,:]
        
        delta_x = (-self.ax * x * self.error_coupling) * self.tau 
        
        psi = torch.exp(-self.h * (x[0,0] - self.c)**2)
        
        front_term = x * (goal - y0)
        
        psi_sum = torch.sum(psi)
        
        psi = psi.unsqueeze(0).unsqueeze(-1).repeat(y.size(0),1,1)
                
        dot_prod = torch.bmm(self.w,psi).squeeze()
        
        f = (front_term*dot_prod*1000)/psi_sum
        
        dd_traj = (self.ay *
                           (self.by * (goal - traj) -
                           d_traj/self.tau) + f) * self.tau
                   
        delta_d_traj = dd_traj * self.tau * self.error_coupling
        new_d_traj = d_traj + delta_d_traj*self.dt  
        delta_traj = new_d_traj * self.error_coupling
             
        ret_tensor = torch.cat([delta_x,torch.zeros_like(goal),torch.zeros_like(y0),delta_d_traj,delta_traj],dim=1)
        
        return ret_tensor       
    
    
    
#class ModelD(nn.Module):
#    def __init__(self,n_output):
#        super(ModelD, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
#        self.bn1 = nn.BatchNorm2d(32)
#        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
#        self.bn2 = nn.BatchNorm2d(64)
#        self.fc1  = nn.Linear(64*28*28, 1024)
#        self.fc2 = nn.Linear(1024, 1024)
#        self.fc3 = nn.Linear(1024, n_output)
#
#    def forward(self, x):
#        batch_size = x.size(0)
#        x = x.view(batch_size, 1, 28,28)
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = F.relu(x)
#        x = self.conv2(x)
#        x = self.bn2(x)
#        x = F.relu(x)
#        x = x.view(batch_size, 64*28*28)
#        x = self.fc1(x)
#        x = F.relu(x)
#        x = self.fc2(x)
#        x = F.relu(x)
#        x = self.fc3(x)
#        x[:,:4] = torch.sigmoid(x[:,:4])
#        x[:,4:] = torch.tanh(x[:,4:])
#        return x   
    
    
    
    
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        #grid = F.affine_grid(theta, x.size())
        #x = F.grid_sample(x, grid)

        return theta

    def forward(self, x):
        # transform the input
        theta = self.stn(x)
        
        return theta    
    
    
class ModelD(nn.Module):
    def __init__(self,n_output):
        super(ModelD, self).__init__()
        #self.spatTr = SpatialTransformer()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1b  = nn.Linear(64*5*5, 500)
        self.fc2b = nn.Linear(500, 10)
        
        self.fc1a  = nn.Linear(64*5*5, 1024)
        self.fc2a = nn.Linear(1024, 512)
        self.fc3a = nn.Linear(512, 10)
        self.fc4a = nn.Linear(20, 512)
        self.fc5a = nn.Linear(512, 1024)
        self.fc6a = nn.Linear(1024, n_output)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28,28)
        #theta = self.spatTr(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(batch_size, 64*5*5)#64*28*28)
        
        xb = self.fc1b(x)
        xb = F.relu(xb)
        xb1 = self.fc2b(xb)
        xb = F.log_softmax(xb1, dim=1)
        
        xa = self.fc1a(x)
        xa = F.relu(xa)
        xa = self.fc2a(xa)
        xa = F.relu(xa)
        xa = self.fc3a(xa)
        xa = F.relu(xa)
        xa = self.fc4a(torch.cat([xa,F.relu(xb1)],dim=1))
        xa = F.relu(xa)
        xa = self.fc5a(xa)
        xa = F.relu(xa)
        xa = self.fc6a(xa)
        xa[:,:4] = torch.sigmoid(xa[:,:4])
        xa[:,4:] = torch.tanh(xa[:,4:])
        
        
        
        return xa,xb#,theta       
    
    
    
    