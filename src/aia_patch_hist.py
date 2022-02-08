#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt

file_list = []
patch_path = '/d1/patches/trn/'
for root,dirs,files in os.walk(patch_path):
	for file in files:
			if file.startswith('aia') and len(file)==20:
				file_list.append(file)

file_list = sorted(file_list)
#print(file_list[:1])
l = []
for i in range(196):
	aia = pickle.load(open(patch_path+file_list[i], "rb" ))
	l = l + list(aia.flatten())
plt.hist(l,bins =20)
plt.yscale('log')
plt.show()
print(len(l))
print(np.mean(l))
print(np.median(l))