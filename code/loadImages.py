#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:51:53 2018

@author: veysiyildiz
"""
from keras.preprocessing import image
import numpy as np
import os.path

def readImages(image_names, path_to_folder):
    '''
    read images in the image_names list, write them to an array. if an image does not exist in the folder 
    a warning will be printed and that part of the array will be ZEROS
        inputs: 
            image_names: a python list contains the names of the images
            path_to_folder: path to the folder which contains the images
        outputs: 
            images: an array of shape (n,w,l,c). n= number of images, w=width(480), l=length(640), c= channels(3)
            
    '''
    if path_to_folder[-1]!='/' : path_to_folder=path_to_folder+'/'
    n=len(image_names)
    
    all_data=np.zeros((n,480,640,3))
    for idx,image_name in enumerate(image_names):
        image_path = path_to_folder + str(image_name)   
        if os.path.isfile(image_path):
            all_data [idx,:,:,:] = np.array(image.load_img(image_path))
        else :
            print image_path,  ' is not a file. **** zeros used instead of image data.'
    
    return all_data
            
        
        
    