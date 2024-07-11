#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:38:52 2024

@author: sjohn
"""

import os
from PIL import Image
import numpy as np
import cv2

os.chdir("/home/sjohn/projects/data-visuals/weather_map")
folder_path = './maps/'

# List all files in the folder
files = os.listdir(folder_path)

# Filter the list to include only PNG files
png_files = [file for file in files if file.endswith('.png')]

sort_files = sorted(png_files)

no_days=7
nf=7*24

subset_files=sort_files[-nf:]


def convert_pictures_to_video(subset_files, pathOut, fps, time):
    ''' this function converts images to video'''
    frame_array=[]
    for i in range (len(subset_files)):
        '''reading images'''
        im='./maps/'+subset_files[i]
        pil_image =Image.open(im).convert('RGB')
        height, width = pil_image.size
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        img = open_cv_image[:, :, ::-1].copy() 
        size=(height,width)
        for k in range (time):
            frame_array.append(img)
    out=cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*"avc1"), fps,size)
    for jj in range(len(frame_array)):
        out.write(frame_array[jj])
    out.release()

# Example:

pathOut='./maps/'+"weather_map.mp4"
fps=1
time=1 # the duration of each pictu
convert_pictures_to_video(subset_files, pathOut, fps, time)


pathOut='/var/www/html/displaywall/files/'+"weather_map.mp4"
fps=1
time=1 # the duration of each pictu
convert_pictures_to_video(subset_files, pathOut, fps, time)
