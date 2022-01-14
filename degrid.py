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
    eitp9 = np.zeros(9)	
    expanded_img = np.zeros((xsize+42*2 ,ysize+42*2))
    expanded_img[42:xsize+42, 42:ysize+42] = grid
    expanded_img[0:42,42:ysize+42] = np.flip(grid[0:42,:],0)
    expanded_img[xsize+42:,42:ysize+42]=np.flip(grid[xsize-42:,:],1)
    expanded_img[42:xsize+42,ysize+42:]=np.flip(grid[:,ysize-42:],0)
    expanded_img[42:xsize+42,0:42]=np.flip(grid[:,0:42],1)
    for j in range(42,ysize+42):
        for i in range(42,xsize+42):
            eitp9[0] = expanded_img[i-42,j]
            eitp9[5] = expanded_img[i,j-42]
            eitp9[1] = expanded_img[i-21,j]
            eitp9[6] = expanded_img[i,j-21]
            eitp9[2] = expanded_img[i,j]
            eitp9[3] = expanded_img[i+21,j]
            eitp9[4] = expanded_img[i,j+21]
            eitp9[7] = expanded_img[i+42,j]
            eitp9[8] = expanded_img[i,j+42]
            grid[i-42,j-42] = np.median(eitp9)

    edija = 10**(image-grid)
    return edija








