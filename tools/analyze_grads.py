#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:26:13 2024

@author: boyuantian
"""

import torch

import matplotlib.pyplot as plt

grads = torch.load("../attrs.pt")



# gs_idx = -3
attr_idx = -1 # 0: xyz, f_dc, f_rest, opacity, scaling, rotation

# xyz and rotation are relatively stable
# scaling, f_dc, and opacity are extremely unstable
                                                                                                               
       
def plotGS(gs_idx):
    val_x, val_y, val_z, val_w = [], [], [], []

    # grads: frame -> attributes -> gaussians -> dimensions
    for idx_frame, frame in enumerate(grads):
        if len(frame):
            print (idx_frame, frame[0][0][0])
            print (frame[0].shape)
            # print (frame[attr_idx])

            # print (frame[attr_idx][gs_idx])

            # print (frame[attr_idx][gs_idx][0])

            # print (frame[attr_idx][gs_idx][1])
            
            
            # for f_dc only
            # val_x.append(frame[attr_idx][gs_idx][0][0].detach().cpu())
            # val_y.append(frame[attr_idx][gs_idx][0][1].detach().cpu())
            # val_z.append(frame[attr_idx][gs_idx][0][2].detach().cpu())
            
            # for rotation only
            # val_x.append(frame[attr_idx][gs_idx][0].detach().cpu())
            # val_y.append(frame[attr_idx][gs_idx][1].detach().cpu())
            # val_z.append(frame[attr_idx][gs_idx][2].detach().cpu())
            # val_w.append(frame[attr_idx][gs_idx][3].detach().cpu())
            
            val_x.append(frame[attr_idx][gs_idx][0].detach().cpu())
            val_y.append(frame[attr_idx][gs_idx][1].detach().cpu())
            val_z.append(frame[attr_idx][gs_idx][2].detach().cpu())
            
    plt.plot(range(len(val_x)), val_x)
    plt.plot(range(len(val_y)), val_y)
    plt.plot(range(len(val_z)), val_z)
    # plt.plot(range(len(val_w)), val_w)
    plt.savefig("attrs/%s.png" % gs_idx)
    plt.close()
    
for i in range(grads[0][0].shape[0]-1, -1, -1):
# for i in range(grads[0][0].shape[0]):
    plotGS(i)
# plotGS(0)
