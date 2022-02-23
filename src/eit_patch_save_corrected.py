#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
import os

import numpy as np
import sunpy.map
import pandas as pd
#import tensorflow as tf
import pickle

# fd_exposure_trn, fd_exposure_val = {'date':[],'exptime':[]},{'date':[],'exptime':[]}

# fd_path_trn = '/d1/fd/trn/eit_171/'
# fd_path_val = '/d1/fd/val/eit_171/'

# for root,dirs,files in os.walk(fd_path_trn):
# 	for file in files:
# 		e = sunpy.map.Map(fd_path_trn+file)
# 		fd_exposure_trn['date'].append(file[3:11])
# 		fd_exposure_trn['exptime'].append(e.meta['exptime'])

# for root,dirs,files in os.walk(fd_path_val):
# 	for file in files:
# 		e = sunpy.map.Map(fd_path_val+file)
# 		fd_exposure_val['date'].append(file[3:11])
# 		fd_exposure_val['exptime'].append(e.meta['exptime'])


# df1 = pd.DataFrame(fd_exposure_trn)
# df2 = pd.DataFrame(fd_exposure_val)
# print(df1)
# print(df2)
# df1.to_csv('/d1/fd/eit_date_exposure_trn.csv')
# df2.to_csv('/d1/fd/eit_date_exposure_val.csv')

df1 = pd.read_csv('/d1/fd/eit_date_exposure_trn.csv')
df2 = pd.read_csv('/d1/fd/eit_date_exposure_val.csv')
print(df1)
print(df2)
def eit_patch_corrector(patch_path,fd_exposure):
	idx = list(fd_exposure['date']).index(int(patch_path[20:28]))
	exptime = fd_exposure['exptime'][idx]
	eit = pickle.load(open(patch_path, "rb" ))
	eit[:,:,0] = eit[:,:,0]*1000*exptime
	return eit


train_path = '/d1/patches/trn/'
patches_trn = []
for root,dirs,files in os.walk(train_path):
	for file in files:
		if file.startswith('eit'):
			patches_trn.append(train_path+file)


val_path = '/d0/patches/val/'
patches_val = []
for root,dirs,files in os.walk(val_path):
	for file in files:
		if file.startswith('eit'):
			patches_val.append(val_path+file)

print(len(patches_trn))
print(len(patches_val))
print(patches_trn[0][20:28])
print(patches_val[0][20:28])

for i in range(len(patches_trn)):
	print(i)
	#eit = pickle.load(open(patches_trn[i], "rb" ))
	#print(np.max(eit[:,:,0]))
	#print(list(df1['date']))
	eit = eit_patch_corrector(patches_trn[i],df1)
	print(np.max(eit[:,:,0]))
	pickle.dump(eit,open(patches_trn[i],'wb'))
	#idx = list(df1['date']).index(int(patches_trn[i][20:28]))
    # print(idx)
	#print(df1['exptime'][idx]) 


for i in range(len(patches_val)):
	print(i)
	#eit = pickle.load(open(patches_trn[i], "rb" ))
	#print(np.max(eit[:,:,0]))
	#print(list(df1['date']))
	eit = eit_patch_corrector(patches_val[i],df2)
	print(np.max(eit[:,:,0]))
	pickle.dump(eit,open(patches_val[i],'wb'))
