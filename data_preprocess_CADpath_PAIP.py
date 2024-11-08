#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 06 15:02:03 2021
@author: spoliveira
INESCTEC
"""
'''CADpath data preprocessing - PAIP dataset [test]'''

import os
import sys
import csv
import numpy as np
import pandas as pd

import time
from datetime import timedelta

from openslide import *
from openslide.deepzoom import DeepZoomGenerator

# from image_view import img_view
from matplotlib import pyplot as plt

# hyperparameters
slide_type = '.svs'
patch_size = 512 #224, 256, 512
overlap = 0
tissue_thresholds = [1]

# Define data directories, list .svs files, load training labels
svs_dir = '/imp-results/PAIP_CRC/slides'
msk_dir = '/imp-results/PAIP_CRC/otsu'

files = np.array([f for f in os.listdir(svs_dir) if f.endswith(slide_type)])

save_csv = '/imp-results/PAIP_CRC'
tiles_datafile = os.path.join(save_csv, 'PAIP_tiles_' + str(patch_size) + '_tissuethr_100.csv')

lst = [['{}_ntiles'.format(int(x*100)), '{}_tiles_grid'.format(int(x*100)), '{}_tiles_coords'.format(int(x*100)), '{}_gleason'.format(int(x*100)), '{}_gleason_soft'.format(int(x*100))] for x in tissue_thresholds]
col = ['slide_name'] + [l for y in lst for l in y]

downfactor = 32

with open(tiles_datafile, 'w') as csvfile:
    w = csv.DictWriter(csvfile, fieldnames=col)
    w.writeheader()
    
    error_list = []
    for ii in range(len(files)):
        
        start = time.time()
        case_id = files[ii].split(slide_type)[0]

        if os.path.isfile(os.path.join(msk_dir, case_id + '_otsu.png')):
            msk_data = Image.open(os.path.join(msk_dir, case_id + '_otsu.png'))

            if msk_data.mode != 'RGB':
                msk_data = msk_data.convert('RGB')

        else:
            # df = df.append(file_tiles, ignore_index=True)
            # df.to_csv(tiles_datafile, index=False)
            print('no otsu mask')
            continue

        print(f"\n{ii+1}/{len(files)} File ID: {case_id}")
    
        file_tiles = {}
        file_tiles['slide_name'] = case_id
       
        for threshold in tissue_thresholds:
            file_tiles['{}_ntiles'.format(int(threshold*100))] = 0
            file_tiles['{}_tiles_grid'.format(int(threshold*100))] = []
            file_tiles['{}_tiles_coords'.format(int(threshold*100))] = []
            file_tiles['{}_gleason'.format(int(threshold*100))] = []
            file_tiles['{}_gleason_soft'.format(int(threshold*100))] = []

        # Open .svs files with openslide
        slide = open_slide(os.path.join(svs_dir, case_id + slide_type))
        slide_dims = slide.dimensions
        small_dims = tuple(round(x/downfactor) for x in slide_dims)
        
        # Generate tiles for .svs & thumbnail files
        tile_size = patch_size - 2 * overlap  
        tiles_svs = DeepZoomGenerator(slide, tile_size, overlap)
        tiles_msk = DeepZoomGenerator(ImageSlide(msk_data), np.round(tile_size/downfactor), overlap) 
        
        max_level_svs = tiles_svs.level_count - 1    
        max_level_msk = tiles_msk.level_count - 1 
    
        # Tiles sorting & labeling
        x, y = 0, 0
        x_tiles, y_tiles = tiles_msk.level_tiles[max_level_msk]
    
        while y < y_tiles:
            sys.stdout.write("\r" + '.... Reading {}/{} slide grid lines'.format(y, y_tiles))  
           
            while x < x_tiles:   
                new_tile_msk = np.array(tiles_msk.get_tile(max_level_msk, (x, y)), dtype=np.uint8)
                
                if new_tile_msk.shape == (patch_size/downfactor, patch_size/downfactor, 3):
                    
                    for threshold in tissue_thresholds:
                        background = (np.sum(new_tile_msk, axis=2) == 0).sum()/(patch_size/downfactor)**2
                        
                        if background <= 1-threshold:    
                            new_tile_svs = np.array(tiles_svs.get_tile(max_level_svs, (x, y)), dtype=np.uint8)

                            if new_tile_svs.shape == (patch_size, patch_size, 3):

                                tile_coords = tiles_svs.get_tile_coordinates(max_level_svs, (x, y))[0]
                                
                                file_tiles['{}_ntiles'.format(int(threshold*100))] += 1
                                file_tiles['{}_tiles_grid'.format(int(threshold*100))].append((y,x))
                                file_tiles['{}_tiles_coords'.format(int(threshold*100))].append(tile_coords)
                                
                x += 1              
            y += 1
            x = 0
        
        slide.close()
        w.writerow(file_tiles)
        slide_time = timedelta(seconds=int(round(time.time()-start)))
        print('\n',slide_time)