#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
def degrid(path):
    import numpy as np
    import cv2
    import sunpy.map
    from skimage.filters.rank import median
    from skimage.util import img_as_float
    from skimage.util import img_as_int
    eit = sunpy.map.Map(path)
    image = eit.data
    sz = image.shape
    xsize = sz[0] 
    ysize = sz[1]
    image=np.log10(np.clip(image,1,np.max(image)))
    eitmoy =  cv2.blur(image,(23,23))
    maximage = np.max(eitmoy) * 0.875		
    subs = np.where(eitmoy > maximage)
    eitmoy[subs] = image[subs]		
    eitmoy =  cv2.blur(eitmoy,(13,13))
    grid  = image-eitmoy
    mn = np.min(grid)
    mx = np.max(grid)
    grid = (grid-mn)/(mx-mn)
    grid = img_as_int(grid)			
    eitp9 = np.zeros(9)	
    expanded_img = np.zeros((xsize+42*2 ,ysize+42*2),dtype=np.int16)
    expanded_img[42:xsize+42, 42:ysize+42] = grid
    expanded_img[0:42,42:ysize+42] = np.flip(grid[0:42,:],0)
    expanded_img[xsize+42:,42:ysize+42]=np.flip(grid[xsize-42:,:],1)
    expanded_img[42:xsize+42,ysize+42:]=np.flip(grid[:,ysize-42:],0)
    expanded_img[42:xsize+42,0:42]=np.flip(grid[:,0:42],1)
    ##median filter mask    
    filter_m = np.zeros((85,85),dtype=np.int16)
    filter_m[0,42] = 1
    filter_m[21,42] = 1
    filter_m[42,42] = 1
    filter_m[63,42] = 1
    filter_m[84,42] = 1
    filter_m[42,0] = 1
    filter_m[42,21] = 1
    filter_m[42,42] = 1
    filter_m[42,63] = 1
    filter_m[42,84] = 1

    sm_img = median(expanded_img,filter_m)
    grid = sm_img[42:xsize+42,42:ysize+42]
    grid = img_as_float(grid)*(mx-mn) + mn
    edija = 10**(image-grid)
    return edija








