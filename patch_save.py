#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""

import tensorflow as tf
import os
path = '/d0/'
import pickle 

import matplotlib.pyplot as plt
from reproject_eit_aia_new import reproject_eit_aia_new
from patch_generate import patch_generate


file_path = '/d1/fd/val/'
files_e = []
files_a = []
dir_e = file_path + 'eit_171/'
dir_a = file_path + 'aia_171/'
for root,dirs,files in os.walk(dir_e):
    for file in files:
        files_e.append(file)
      
for root,dirs,files in os.walk(dir_a):
    for file in files:
        files_a.append(file)

file_list_aia = sorted(files_a)
file_list_eit = sorted(files_e)


for n in range(0,len(file_list_eit)):
  file_e = dir_e+file_list_eit[n]
  file_a = dir_a+file_list_aia[n]
  eit_lr,eit_hr,mask,aia = reproject_eit_aia_new(file_e,file_a)
  patches_e,patches_a,patches_na,titl = patch_generate(eit_lr,eit_hr,mask,aia,16,0)
  patches_es,patches_as,patches_nas,titl = patch_generate(eit_lr,eit_hr,mask,aia,16,1)
  name_e = 'eit_'+file_list_eit[n][3:11]
  name_a = 'aia_'+ file_list_aia[n][14:18] + file_list_aia[n][19:21] + file_list_aia[n][22:24] 
  sh = patches_e.shape
  sh1 = patches_es.shape
  for i in range(sh[0]):
   # pickle.dump(patches_e[i,:,:,:], open( path+"patches/val/"+name_e+'_0_'+str(i).zfill(3)+".p", "wb" ))
    pickle.dump(patches_a[i,:,:,0], open( path+"patches/val/"+name_a+'_0_'+str(i).zfill(3)+".p", "wb" ))
   # pickle.dump(patches_na[i,:,:,0], open( path+"patches/val/"+name_a+'_n_0_'+str(i).zfill(3)+".p", "wb" ))
  for i in range(sh1[0]):
   # pickle.dump(patches_es[i,:,:,:], open( path+"patches/val/"+name_e+'_1_'+str(i).zfill(3)+".p", "wb" ))
    pickle.dump(patches_as[i,:,:,0], open( path+"patches/val/"+name_a+'_1_'+str(i).zfill(3)+".p", "wb" ))
   # pickle.dump(patches_nas[i,:,:,0], open( path+"patches/val/"+name_a+'_n_1_'+str(i).zfill(3)+".p", "wb" ))
  print(str((n+1)) + ' out of '+str(len(file_list_eit))+ ' finished')







