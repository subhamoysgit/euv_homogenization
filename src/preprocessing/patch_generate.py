#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""

def patch_generate(eit_lr,eit_hr,mask,aia,grid_size,shift):
	import math 
	import numpy as np
	import cv2
	eit_orig = eit_lr.copy()
	aia_orig = aia.copy()
	eit_blown = eit_hr.copy()
	aia_blown = aia_orig.copy()
	kernel = mask.copy()<1
	prof = cv2.resize(mask,(1024,1024))
	eit_blown_ = np.asarray(eit_blown, dtype = 'float32')
	aia_blown_ = np.asarray(aia_blown, dtype = 'float32')
	eit_blown[kernel==0]=0
	aia_blown[kernel==0]=0
	eit_blown = np.asarray(eit_blown, dtype = 'float32')
	aia_blown = np.asarray(aia_blown, dtype = 'float32')
	aia_blown = cv2.blur(aia_blown,(4, 4))
	k=0
	# x_c = []
	titl = []
	patch_size = 4096/grid_size 
	patches_e = np.zeros((int(grid_size**2),int(1024//grid_size),int(1024//grid_size),2))
	patches_a = np.zeros((int(grid_size**2),int(4096//grid_size),int(4096//grid_size),1))
	patches_na = np.zeros((int(grid_size**2),int(4096//grid_size),int(4096//grid_size),1))
	for i in range(grid_size**2):
		y0 = int(patch_size*(i % grid_size)) + int(0.5*patch_size)*shift
		x0 = int(patch_size*(i//grid_size)) + int(0.5*patch_size)*shift
		x1 = int(x0+patch_size)
		y1 = int(y0+patch_size)
		if x1<=4096 and y1<=4096:
			template = eit_blown[y0:y1,x0:x1]
			patch_margin = int(0.125*patch_size)
			w, h = template.shape[::-1]
			xl = np.max([x0-patch_margin ,0])
			xh = np.min([x1+patch_margin ,4096])
			yl = np.max([y0-patch_margin ,0])
			yh = np.min([y1+patch_margin ,4096])
			if math.isnan(np.sum(aia_blown_[yl:yh,xl:xh]))==False:
				if np.sum(template)>0:
						method = 'cv2.TM_CCOEFF_NORMED'
						img = aia_blown.copy()
						method = eval(method)
						img_crop = img[yl:yh,xl:xh]
						res = cv2.matchTemplate(np.log(1+img_crop),np.log(1+template),method)
						min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
						top_left = (max_loc[0] + xl, max_loc[1] + yl)
						bottom_right = (top_left[0] + w, top_left[1] + h)
						patches_e[k,:,:,0] = eit_orig[int(y0//4):int(y1//4),int(x0//4):int(x1//4)]
						patches_e[k,:,:,1] = prof[y0//4:y1//4,x0//4:x1//4]
						patches_a[k,:,:,0] = aia_blown_[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
						patches_na[k,:,:,0] = aia_blown_[y0:y1,x0:x1]
						titl.append('with')
						k = k+1
						# x_c.append([max_loc[0],max_loc[1]])
			else:
				patches_e[k,:,:,0] = eit_orig[int(y0//4):int(y1//4),int(x0//4):int(x1//4)]
				patches_e[k,:,:,1] = prof[y0//4:y1//4,x0//4:x1//4]
				patches_a[k,:,:,0] = aia_blown_[y0:y1,x0:x1]
				patches_na[k,:,:,0] = aia_blown_[y0:y1,x0:x1]
				titl.append('w/o')
				k = k+1

	patches_e = patches_e[:k,:,:,:]
	patches_a = patches_a[:k,:,:,:]
	patches_na = patches_na[:k,:,:,:]
	return patches_e,patches_a,patches_na,titl   #img_crop,res,x_c




