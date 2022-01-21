#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np
import os
from keras import backend as K
import cv2
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.callbacks import ModelCheckpoint
from skimage import img_as_ubyte
from keras.models import *
from keras.layers import *
from keras.optimizers import *
seed_value = 5421
from keras.engine.topology import Layer
from keras.engine import InputSpec


from scipy import stats
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
axis_x = 10**np.linspace(-1,1,20)
axis_y = 10**np.linspace(-1,1,20)
axis_yy = 10**np.linspace(-3,1,40)
X, Y = np.meshgrid(axis_x, axis_y)
XX, YY = np.meshgrid(axis_x, axis_yy)
ax_val = 0.5*(axis_x[:19]+axis_x[1:])
ax_val_y = 0.5*(axis_yy[:39]+axis_yy[1:])
bins_y = axis_y[1:] - axis_y[:19]
bins_yy = axis_yy[1:] - axis_yy[:39]
path = "/home/subhamoy/mycodes/patch_code/"
names_1 = ["hist2_all.p","hist1.p","hist1_lossMH_v3.p","hist1_lossMS_v4.p","hist1_lossMG_v3.p"]
names_2 = ["hist1_4by4_lossM.p","hist1_4by4_lossMH.p","hist1_4by4_lossMH_v2.p","hist1_4by4_lossMH_v3.p","hist1_4by4_lossMH_v4.p","hist1_4by4_lossMS.p","hist1_4by4_lossMS.p","hist1_4by4_lossMS.p","hist1_4by4_lossMS.p","hist1_4by4_lossMS.p","hist1_4by4_lossMG.p","hist1_4by4_lossMG_v2.p","hist1_4by4_lossMG_v3.p","hist1_4by4_lossMG_v3.p"] 
names_3 = ["ssim1.p","ssim1_lossMH.p","ssim1_lossMH_v2.p","ssim1_lossMH_v3.p","ssim1_lossMH_v3.p","ssim1_lossMS.p","ssim1_lossMS.p","ssim1_lossMS.p","ssim1_lossMS.p","ssim1_lossMS.p","ssim1_lossMG.p","ssim1_lossMG.p","ssim1_lossMG.p","ssim1_lossMG.p"]
titles = ["Upsampling, scaling","MSE (Baseline)", "MSE + 20*Hist","MSE - 0.01*SSIM","MSE + 400*Grad"]
Clr = [(0.00, 0.00, 0.00),
      (0.31, 0.24, 0.00),
      (0.43, 0.16, 0.49),
      (0.32, 0.70, 0.30),
      (0.45, 0.70, 0.90),
      (1.00, 0.82, 0.67)]



axis_xx = np.linspace(-1,1,20)
ax_vall = 0.5*(axis_xx[:19]+axis_xx[1:])


fig, ax = plt.subplots(3,len(titles),figsize=(15*len(titles),15*3),sharex = True)
axs = ax.ravel()
#hist11_den_y = np.zeros((19,19))
#hist1_den_y = np.zeros((19,19))
for n in range(len(titles)): 
  hist11 = pickle.load(open( path + names_1[n], "rb" ))
  hist1 = pickle.load(open( path + names_2[n], "rb" ))
  ssim1 = pickle.load(open( path + names_3[n], "rb" ))
  ssim1_den,be1 = np.histogram(ssim1,bins=axis_xx, density=True)
  hist11_den_y = np.zeros((19,19))
  hist1_den_y = np.zeros((39,19))  
  for j in range(19):
    hist11_den_y[:,j] = hist11[:,j]/(np.sum(hist11[:,j])*bins_y[:])
    hist1_den_y[:,j] = hist1[:,j]/(np.sum(hist1[:,j])*bins_yy[:]) 
  expect_11 = np.zeros(19)
  stdev_11 = np.zeros(19)
  expect_1 = np.zeros(19)
  stdev_1 = np.zeros(19)
  print(np.max(hist1_den_y))
  print(np.max(hist11_den_y))
  for i in range(19):
    expect_11[i] = np.dot(hist11[:,i],ax_val)/np.nansum(hist11[:,i])
    stdev_11[i] = (np.dot((ax_val-expect_11[i])**2,hist11[:,i])/np.nansum(hist11[:,i]))**0.5
    expect_1[i] = np.dot(hist1[:,i],ax_val_y)/np.nansum(hist1[:,i])
    stdev_1[i] = (np.dot((ax_val_y-expect_1[i])**2,hist1[:,i])/np.nansum(hist1[:,i]))**0.5


  if n==0:
    axs[n].pcolormesh(X, Y,hist11_den_y**0.5,vmin=0,vmax=1,cmap=cm.get_cmap('Greys'));axs[n].axis('square')#;plt.xscale('log');plt.yscale('log')
    axs[n].plot(ax_val,expect_11,linestyle='--',  color=Clr[2], linewidth=1.2,label = 'expected')
    axs[n].fill_between(ax_val, expect_11 - stdev_11, expect_11 + stdev_11, fc=Clr[2], alpha=0.3, ec='None')
    axs[n].plot([0.1,10], [0.1,10], color='#00dd66', linestyle=':', linewidth=1.2,label = 'y=x')
    axs[n].set_xlim([0.1,10])
    axs[n].set_xticks([])
    axs[n].set_title(titles[n],fontsize=12)
    axs[n].set_ylabel('AIA target',fontsize=12)
    axs[n].legend(frameon = False)

    axs[n+len(titles)].plot(ax_val,100*(expect_11-ax_val)/ax_val,linestyle='--',  color=Clr[2], linewidth=1.2)
    axs[n+len(titles)].fill_between(ax_val, 0*ax_val, 100*(expect_11 - ax_val)/ax_val, fc=Clr[2], alpha=0.8, ec='None')
    axs[n+len(titles)].plot([0.1,10],[5,5],linestyle=':',  color='r', linewidth=1.2)
    axs[n+len(titles)].plot([0.1,10],[-5,-5],linestyle=':',  color='r', linewidth=1.2)    
    axs[n+len(titles)].plot([0.1,10], [0,0], color='#00dd66', linestyle=':', linewidth=1.2)
    axs[n+len(titles)].set_xlim([0.1,10])
    axs[n+len(titles)].set_ylim([-20,20])
    axs[n+len(titles)].set_xticks([])
    axs[n+len(titles)].set_yticks([-15,-10,-5,0,5,10,15])
    axs[n+len(titles)].set_aspect(9.9/40)
    axs[n+len(titles)].set_ylabel('% Relative Residual',fontsize=12)

    axs[n+2*len(titles)].fill_between(ax_val, 0*ax_val, 100*stdev_11/expect_11, fc=Clr[2], alpha=0.8, ec='None')
    axs[n+2*len(titles)].plot([0.1,10], [0,0], color='#00dd66', linestyle=':', linewidth=1.2)
    axs[n+2*len(titles)].set_xlim([0.1,10])
    axs[n+2*len(titles)].set_ylim([-10,100])
    axs[n+2*len(titles)].set_xticks([0,2,4,6,8])
    axs[n+2*len(titles)].set_yticks([0,20,40,60,80])
    axs[n+2*len(titles)].set_aspect(9.9/110)
    axs[n+2*len(titles)].set_ylabel('% Relative Variance',fontsize=12)
    axs[n+2*len(titles)].set_xlabel('AIA Inferred',fontsize=12)

  elif n==1:
    mu = expect_11
    sd = stdev_11
    mu_1 = expect_1
    s1 = ssim1_den
    axs[n].pcolormesh(X, Y,hist11_den_y**0.5,vmin=0,vmax=1,cmap=cm.get_cmap('Greys'));axs[n].axis('square')#;plt.xscale('log');plt.yscale('log')
    axs[n].plot(ax_val,expect_11,linestyle='--',  color=Clr[2], linewidth=1.2,label = 'expected')
    axs[n].fill_between(ax_val, expect_11 - stdev_11, expect_11 + stdev_11, fc=Clr[2], alpha=0.3, ec='None')
    axs[n].plot([0.1,10], [0.1,10], color='#00dd66', linestyle=':', linewidth=1.2,label = 'y=x')
    axs[n].set_xlim([0.1,10])
    axs[n].set_xticks([])
    axs[n].set_yticks([])
    axs[n].set_title(titles[n],fontsize=12)
    axs[n].legend(frameon = False)

    axs[n+len(titles)].plot(ax_val,100*(expect_11-ax_val)/ax_val,linestyle='--',  color=Clr[2], linewidth=1.2)
    axs[n+len(titles)].fill_between(ax_val, 0*ax_val, 100*(expect_11 - ax_val)/ax_val, fc=Clr[2], alpha=0.8, ec='None')
    axs[n+len(titles)].plot([0.1,10],[5,5],linestyle=':',  color='r', linewidth=1.2)
    axs[n+len(titles)].plot([0.1,10],[-5,-5],linestyle=':',  color='r', linewidth=1.2)
    axs[n+len(titles)].plot([0.1,10], [0,0], color='#00dd66', linestyle=':', linewidth=1.2)
    axs[n+len(titles)].set_xlim([0.1,10])
    axs[n+len(titles)].set_ylim([-20,20])
    axs[n+len(titles)].set_xticks([])
    axs[n+len(titles)].set_yticks([])
    axs[n+len(titles)].set_aspect(9.9/40)

    axs[n+2*len(titles)].fill_between(ax_val, 0*ax_val, 100*stdev_11/expect_11, fc=Clr[2], alpha=0.8, ec='None')
    axs[n+2*len(titles)].plot([0.1,10], [0,0], color='#00dd66', linestyle=':', linewidth=1.2)
    axs[n+2*len(titles)].set_xlim([0.1,10])
    axs[n+2*len(titles)].set_ylim([-10,100])
    axs[n+2*len(titles)].set_yticks([])
    axs[n+2*len(titles)].set_aspect(9.9/110)
    axs[n+2*len(titles)].set_xlabel('AIA Inferred',fontsize=12)
    axs[n+2*len(titles)].set_xticks([0,2,4,6,8])

  else:
    axs[n].pcolormesh(X, Y,hist11_den_y**0.5,vmin=0,vmax=1,cmap=cm.get_cmap('Greys'));axs[n].axis('square')#;plt.xscale('log');plt.yscale('log')
    axs[n].plot(ax_val,expect_11,linestyle='--',  color='#ffa232', linewidth=1.2,label = 'expected')
    axs[n].plot(ax_val,mu,linestyle='--',  color=Clr[2], linewidth=1.2)
    axs[n].fill_between(ax_val, expect_11 - stdev_11, expect_11 + stdev_11, fc='#ffa232', alpha=0.3, ec='None')
    axs[n].plot([0.1,10], [0.1,10], color='#00dd66', linestyle=':', linewidth=1.2,label = 'y=x')
    axs[n].set_xlim([0.1,10])
    axs[n].set_xticks([])
    axs[n].set_yticks([])
    axs[n].set_title(titles[n],fontsize=12)
    axs[n].legend(frameon = False)

    axs[n+len(titles)].plot(ax_val,100*(expect_11-ax_val)/ax_val,linestyle='--',  color='#ffa232', linewidth=1.2)
    axs[n+len(titles)].plot(ax_val,100*(mu-ax_val)/ax_val,linestyle='--',  color=Clr[2], linewidth=1.2)
    axs[n+len(titles)].plot([0.1,10],[5,5],linestyle=':',  color='r', linewidth=1.2)
    axs[n+len(titles)].plot([0.1,10],[-5,-5],linestyle=':',  color='r', linewidth=1.2)
    axs[n+len(titles)].fill_between(ax_val, 0*ax_val, 100*(expect_11 - ax_val)/ax_val, fc='#ffa232', alpha=0.8, ec='None')
    axs[n+len(titles)].plot([0.1,10], [0,0], color='#00dd66', linestyle=':', linewidth=1.2)
    axs[n+len(titles)].set_xlim([0.1,10])
    axs[n+len(titles)].set_ylim([-20,20])
    axs[n+len(titles)].set_xticks([])
    axs[n+len(titles)].set_yticks([])
    axs[n+len(titles)].set_aspect(9.9/40)

    axs[n+2*len(titles)].fill_between(ax_val, 0*ax_val, 100*stdev_11/expect_11, fc='#ffa232', alpha=0.8, ec='None')
    axs[n+2*len(titles)].plot(ax_val,100*sd/mu,linestyle=':',  color=Clr[2], linewidth=1.2)
    axs[n+2*len(titles)].plot([0.1,10], [0,0], color='#00dd66', linestyle=':', linewidth=1.2)
    axs[n+2*len(titles)].set_xlim([0.1,10])
    axs[n+2*len(titles)].set_ylim([-10,100])
    axs[n+2*len(titles)].set_yticks([])
    axs[n+2*len(titles)].set_xticks([2,4,6,8])
    axs[n+2*len(titles)].set_aspect(9.9/110)
    axs[n+2*len(titles)].set_xlabel('AIA Inferred',fontsize=12)
    
fig.subplots_adjust(wspace=0.0, hspace=0.0)#0.4)
plt.show()


