#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
from scipy import ndimage
import csv
import numpy as np
from sunpy.map import Map
import astropy.units as u
from reproject import reproject_exact, reproject_interp
from sunpy.coordinates import Helioprojective

def reproject_eit_aia(file_e,file_a):

    ####rescaled EIT map
    scale_factor = 4
    EITmap = Map(file_e)
    meta_e = EITmap.meta
    padX = [int(1024 - meta_e['crpix1']), int(meta_e['crpix1'])]
    padY = [int(1024 - meta_e['crpix2']), int(meta_e['crpix2'])]

    EIT_data = EITmap.data
    imgP = np.pad(EIT_data/100, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, meta_e['sc_roll'], reshape=False)
    imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    EIT_data = imgC
    eit_lr = Map(EIT_data,EITmap.meta)
    eit_lr = eit_lr.rotate(recenter=True)
    eit_s = eit_lr.data
    # Pad image, if necessary
    target_shape = int(4096)
    # Reform map to new size if original shape is too small
    new_fov = np.zeros((target_shape, target_shape)) * np.nan
    new_meta = EITmap.meta
    new_meta['crpix1'] = new_meta['crpix1'] - EITmap.data.shape[0] / 2 + new_fov.shape[0] / 2
    new_meta['crpix2'] = new_meta['crpix2'] - EITmap.data.shape[1] / 2 + new_fov.shape[1] / 2
    # Identify the indices for appending the map original FoV
    i1 = int(new_fov.shape[0] / 2 - EITmap.data.shape[0] / 2)
    i2 = int(new_fov.shape[0] / 2 + EITmap.data.shape[0] / 2)
    # Insert original image in new field of view
    new_fov[i1:i2, i1:i2] = EIT_data[:,:] #EITmap.data[:, :]
    # Assemble Sunpy map
    EITmap = Map(new_fov, new_meta)
    EITmap = EITmap.rotate(scale=scale_factor, recenter=True)
    sz_x_diff = (EITmap.data.shape[0]-target_shape)//2
    sz_y_diff = (EITmap.data.shape[0]-target_shape)//2
    EITmap.meta['crpix1'] = EITmap.meta['crpix1']-sz_x_diff
    EITmap.meta['crpix2'] = EITmap.meta['crpix2']-sz_y_diff
    EITmap.meta['solar_r'] = scale_factor*EITmap.meta['solar_r']
    EITmap = Map(EITmap.data[sz_x_diff:sz_x_diff+target_shape, sz_y_diff:sz_y_diff+target_shape].copy(), EITmap.meta)
    ###aia map
    date_aia = []
    aia_degrad = []
    ###aia degradation factor determination
    with open('/d1/data_degrad_degrad_171.csv', 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        date_aia.append(row[0][:4]+row[0][5:7]+row[0][8:10])
        aia_degrad.append(row[1])
    aia_date = file_a[33:][0:4] + file_a[33:][5:7] + file_a[33:][8:10]
    f_aia = float(aia_degrad[date_aia.index(aia_date)])

    AIA_map = Map(file_a)
    AIA_map = AIA_map.rotate(recenter=True)
    # # Crop AIA image to desired shape
    sz_x_diff = (AIA_map.data.shape[0]-target_shape)//2
    sz_y_diff = (AIA_map.data.shape[0]-target_shape)//2
    AIA_map.meta['crpix1'] = AIA_map.meta['crpix1']-sz_x_diff
    AIA_map.meta['crpix2'] = AIA_map.meta['crpix2']-sz_y_diff
    AIA_map = Map(AIA_map.data[sz_x_diff:sz_x_diff+target_shape, sz_y_diff:sz_y_diff+target_shape].copy(), AIA_map.meta)


    with Helioprojective.assume_spherical_screen(AIA_map.observer_coordinate,only_off_disk=True):
      output, footprint = reproject_interp(AIA_map, EITmap.wcs, EITmap.data.shape)
    eit_aia_map = Map(output, EITmap.meta)
    x, y = np.meshgrid(*[np.arange(v.value) for v in EITmap.dimensions]) * u.pixel
    hpc_coords = EITmap.pixel_to_world(x, y)
    rSun = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / (EITmap.rsun_obs)
    maskEIT = rSun
    offset = 0
    eit_d = EITmap.data.copy() + offset
    aia_d = eit_aia_map.data.copy()/(1000*AIA_map.meta['exptime']*f_aia)

    return eit_s,eit_d,maskEIT,aia_d



